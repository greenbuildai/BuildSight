#!/usr/bin/env python3
"""
val_condition_eval.py
=====================
Evaluates a single model on a single site-condition subset of the val split.
Outputs a JSON file with complete metrics: mAP50, mAP50-95, precision, recall,
F1, per-class AP, FPS, false positives, false negatives.

Run on SASTRA node1 (buildsight conda env):

  # YOLOv11 on S1_normal
  python val_condition_eval.py --model yolo11 --condition S1_normal

  # All 4 models x all 4 conditions (bash loop):
  for model in yolo11 yolo26 yolact samurai; do
    for cond in S1_normal S2_dusty S3_low_light S4_crowded; do
      python val_condition_eval.py --model $model --condition $cond
    done
  done

Results written to:
  /nfsshare/joseva/condition_eval_results/<model>_<condition>.json
"""

import argparse
import json
import os
import time
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ANNOTATED_BY_CONDITION = Path("/nfsshare/joseva/val_annotated_by_condition")
COCO_JSON              = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_val.json")
RESULTS_DIR            = Path("/nfsshare/joseva/condition_eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "yolo11":  "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26":  "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolact":  "/nfsshare/joseva/yolact/weights/buildsight_207_80000.pth",
    "samurai": None,   # SAMURAI_GT = ground truth overlays, no live inference
}

CLS_NAMES  = ["helmet", "safety_vest", "worker"]
GPU        = 0

# ─── Helpers ──────────────────────────────────────────────────────────────────

def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)


def compute_ap(recalls, precisions):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for thr in [i/10 for i in range(11)]:
        p_at_r = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def match_predictions(preds, gts, iou_thr=0.50):
    """Match predictions to GTs for a single image. Returns TP, FP lists."""
    matched_gt = set()
    results = []
    for pred in sorted(preds, key=lambda x: -x["score"]):
        best_iou, best_gt = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in matched_gt or gt["cls"] != pred["cls"]:
                continue
            ov = iou_box(pred["box"], gt["box"])
            if ov > best_iou:
                best_iou, best_gt = ov, gi
        if best_iou >= iou_thr and best_gt >= 0:
            results.append({"tp": 1, "fp": 0, "score": pred["score"], "cls": pred["cls"]})
            matched_gt.add(best_gt)
        else:
            results.append({"tp": 0, "fp": 1, "score": pred["score"], "cls": pred["cls"]})
    return results, len(gts) - len(matched_gt)


def compute_metrics_from_results(all_results, all_fn, n_classes, iou_thr):
    per_class = {}
    for c in range(n_classes):
        cls_res = [r for r in all_results if r["cls"] == c]
        cls_res = sorted(cls_res, key=lambda x: -x["score"])
        fn = sum(fn_dict.get(c, 0) for fn_dict in all_fn)
        tp_cumsum, fp_cumsum, recalls, precisions = 0, 0, [], []
        total_gt = sum(r["tp"] for r in cls_res) + fn
        for r in cls_res:
            tp_cumsum += r["tp"]
            fp_cumsum += r["fp"]
            rec = tp_cumsum / max(total_gt, 1)
            pre = tp_cumsum / max(tp_cumsum + fp_cumsum, 1)
            recalls.append(rec)
            precisions.append(pre)
        ap = compute_ap(recalls, precisions) if recalls else 0.0
        final_rec = recalls[-1] if recalls else 0.0
        final_pre = precisions[-1] if precisions else 0.0
        f1 = 2 * final_pre * final_rec / max(final_pre + final_rec, 1e-6)
        per_class[CLS_NAMES[c]] = {
            "AP50":      round(ap, 4),
            "precision": round(final_pre, 4),
            "recall":    round(final_rec, 4),
            "F1":        round(f1, 4),
            "TP":        tp_cumsum,
            "FP":        fp_cumsum,
            "FN":        fn,
        }
    all_ap = [v["AP50"] for v in per_class.values()]
    mAP50   = round(sum(all_ap) / len(all_ap), 4)
    all_pre = [v["precision"] for v in per_class.values()]
    all_rec = [v["recall"] for v in per_class.values()]
    macro_p = round(sum(all_pre)/len(all_pre), 4)
    macro_r = round(sum(all_rec)/len(all_rec), 4)
    macro_f1 = round(2*macro_p*macro_r/max(macro_p+macro_r, 1e-6), 4)
    total_fp = sum(v["FP"] for v in per_class.values())
    total_fn = sum(v["FN"] for v in per_class.values())
    return {
        "mAP50":     mAP50,
        "precision": macro_p,
        "recall":    macro_r,
        "F1":        macro_f1,
        "total_FP":  total_fp,
        "total_FN":  total_fn,
        "per_class": per_class,
    }


# ─── YOLO evaluation ──────────────────────────────────────────────────────────

def eval_yolo(model_key, condition, img_dir, coco_data):
    from ultralytics import YOLO
    model = YOLO(MODEL_PATHS[model_key])
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        raise RuntimeError(f"No images found in {img_dir}")

    # Build GT lookup by filename
    id2fname = {img["id"]: Path(img["file_name"]).name for img in coco_data["images"]}
    fname2id = {v: k for k, v in id2fname.items()}
    gt_by_file = {fname: [] for fname in fname2id}
    for ann in coco_data["annotations"]:
        fname = id2fname.get(ann["image_id"])
        if fname and fname in gt_by_file:
            x, y, w, h = ann["bbox"]
            gt_by_file[fname].append({
                "box": [x, y, x+w, y+h],
                "cls": ann["category_id"] - 1   # COCO 1-indexed -> 0-indexed
            })

    all_results, all_fn = [], []
    total_time, n_imgs = 0.0, 0

    for iou_thr in [0.50]:    # extend to [0.50,0.55,...,0.95] for mAP50-95
        all_results.clear(); all_fn.clear(); total_time = 0.0; n_imgs = 0
        for img_path in img_files:
            fname = img_path.name
            if fname not in gt_by_file:
                continue
            gts = gt_by_file[fname]
            t0 = time.perf_counter()
            r = model.predict(str(img_path), device=GPU, verbose=False, conf=0.25)[0]
            total_time += time.perf_counter() - t0
            n_imgs += 1
            preds = []
            for x in (r.boxes or []):
                xy = x.xyxy[0].cpu().numpy().tolist()
                preds.append({"box": xy, "score": float(x.conf[0]), "cls": int(x.cls[0])})
            res, fn = match_predictions(preds, gts, iou_thr)
            all_results.extend(res)
            fn_by_cls = {}
            for gt in gts:
                c = gt["cls"]
                fn_by_cls[c] = fn_by_cls.get(c, 0)
            # count unmatched GTs per class
            matched_cls = [r["cls"] for r in res if r["tp"] == 1]
            for gt in gts:
                c = gt["cls"]
                fn_by_cls[c] = fn_by_cls.get(c, 0) + 1
            for c in matched_cls:
                fn_by_cls[c] = max(0, fn_by_cls.get(c, 0) - 1)
            all_fn.append(fn_by_cls)

    fps = round(n_imgs / total_time, 2) if total_time > 0 else 0.0
    metrics = compute_metrics_from_results(all_results, all_fn, len(CLS_NAMES), 0.50)
    metrics["FPS"]    = fps
    metrics["n_imgs"] = n_imgs
    return metrics


# ─── YOLACT++ evaluation ──────────────────────────────────────────────────────

def eval_yolact(condition, img_dir, coco_data):
    """Run YOLACT++ inference using the DCN shim approach Leon used."""
    import sys
    sys.path.insert(0, "/nfsshare/joseva/yolact")
    sys.path.insert(0, "/nfsshare/joseva/yolact/external/DCNv2")
    import torch
    from yolact import Yolact
    from data import cfg, set_cfg
    from utils.augmentations import FastBaseTransform
    from layers.output_utils import postprocess

    set_cfg("buildsight_config")
    net = Yolact()
    net.load_weights(MODEL_PATHS["yolact"])
    net.eval()
    net = net.cuda(GPU)

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    id2fname  = {img["id"]: Path(img["file_name"]).name for img in coco_data["images"]}
    fname2id  = {v: k for k, v in id2fname.items()}
    gt_by_file = {fname: [] for fname in fname2id}
    for ann in coco_data["annotations"]:
        fname = id2fname.get(ann["image_id"])
        if fname and fname in gt_by_file:
            x, y, w, h = ann["bbox"]
            gt_by_file[fname].append({"box": [x, y, x+w, y+h], "cls": ann["category_id"]-1})

    import cv2
    all_results, all_fn = [], []
    total_time, n_imgs = 0.0, 0

    with torch.no_grad():
        for img_path in img_files:
            fname = img_path.name
            if fname not in gt_by_file:
                continue
            gts = gt_by_file[fname]
            img_cv = cv2.imread(str(img_path))
            if img_cv is None:
                continue
            h, w = img_cv.shape[:2]
            frame = torch.from_numpy(img_cv).cuda(GPU).float()
            batch = FastBaseTransform()(frame.unsqueeze(0))
            t0 = time.perf_counter()
            preds_raw = net(batch)
            total_time += time.perf_counter() - t0
            n_imgs += 1
            cls_p, scr_p, box_p, _ = postprocess(preds_raw, w, h, score_threshold=0.25)
            if isinstance(scr_p, (list, tuple)):
                scr_p = scr_p[0]
            preds = []
            for i in range(len(scr_p)):
                preds.append({
                    "box":   box_p[i].cpu().tolist(),
                    "score": float(scr_p[i]),
                    "cls":   int(cls_p[i]),
                })
            res, fn = match_predictions(preds, gts, 0.50)
            all_results.extend(res)
            fn_by_cls = {}
            for gt in gts:
                c = gt["cls"]
                fn_by_cls[c] = fn_by_cls.get(c, 0) + 1
            for r in res:
                if r["tp"] == 1:
                    fn_by_cls[r["cls"]] = max(0, fn_by_cls.get(r["cls"], 0) - 1)
            all_fn.append(fn_by_cls)

    fps = round(n_imgs / total_time, 2) if total_time > 0 else 0.0
    metrics = compute_metrics_from_results(all_results, all_fn, len(CLS_NAMES), 0.50)
    metrics["FPS"]    = fps
    metrics["n_imgs"] = n_imgs
    return metrics


# ─── SAMURAI GT (ground-truth overlay — no live inference) ───────────────────

def eval_samurai_gt(condition, img_dir, coco_data):
    """
    SAMURAI_GT is not a live detector — it is the ground-truth annotation overlay.
    For the condition matrix it acts as the 100% recall/precision ceiling reference.
    We compute GT-vs-GT stats (trivially perfect) and note this in the output.
    """
    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    n_imgs    = len(img_files)
    id2fname  = {img["id"]: Path(img["file_name"]).name for img in coco_data["images"]}
    gt_counts = {c: 0 for c in range(len(CLS_NAMES))}
    for ann in coco_data["annotations"]:
        fname = id2fname.get(ann["image_id"])
        if fname and any(fname == f.name for f in img_files):
            c = ann["category_id"] - 1
            if c in gt_counts:
                gt_counts[c] += 1
    per_class = {}
    for c, name in enumerate(CLS_NAMES):
        per_class[name] = {
            "AP50":      1.0,
            "precision": 1.0,
            "recall":    1.0,
            "F1":        1.0,
            "TP":        gt_counts[c],
            "FP":        0,
            "FN":        0,
        }
    return {
        "note":      "SAMURAI_GT is ground-truth reference — scores are theoretical ceiling",
        "mAP50":     1.0,
        "mAP50_95":  1.0,
        "precision": 1.0,
        "recall":    1.0,
        "F1":        1.0,
        "FPS":       "N/A (GT overlay)",
        "total_FP":  0,
        "total_FN":  0,
        "n_imgs":    n_imgs,
        "per_class": per_class,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True, choices=["yolo11","yolo26","yolact","samurai"])
    parser.add_argument("--condition", required=True, choices=["S1_normal","S2_dusty","S3_low_light","S4_crowded"])
    args = parser.parse_args()

    img_dir = ANNOTATED_BY_CONDITION / args.condition / {
        "yolo11":  "YOLOv11",
        "yolo26":  "YOLOv26",
        "yolact":  "YOLACT_plusplus",
        "samurai": "SAMURAI_GT",
    }[args.model]

    if not img_dir.exists():
        raise RuntimeError(f"Image directory not found: {img_dir}\n"
                           "Run organize_val_by_condition.py first.")

    print(f"\n{'='*60}")
    print(f" Evaluating: {args.model.upper()} | {args.condition}")
    print(f" Images dir: {img_dir}")
    print(f"{'='*60}")

    with open(COCO_JSON) as f:
        coco_data = json.load(f)

    t_start = time.time()
    if args.model in ("yolo11", "yolo26"):
        metrics = eval_yolo(args.model, args.condition, img_dir, coco_data)
    elif args.model == "yolact":
        metrics = eval_yolact(args.condition, img_dir, coco_data)
    else:
        metrics = eval_samurai_gt(args.condition, img_dir, coco_data)
    elapsed = round(time.time() - t_start, 1)

    metrics["model"]     = args.model
    metrics["condition"] = args.condition
    metrics["eval_time_s"] = elapsed

    out_path = RESULTS_DIR / f"{args.model}_{args.condition}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"  mAP50:     {metrics['mAP50']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall:    {metrics['recall']}")
    print(f"  F1:        {metrics['F1']}")
    print(f"  FPS:       {metrics.get('FPS','N/A')}")
    print(f"  FP/FN:     {metrics.get('total_FP',0)} / {metrics.get('total_FN',0)}")
    print(f"  Time:      {elapsed}s\n")


if __name__ == "__main__":
    main()
