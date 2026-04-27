#!/usr/bin/env python3
"""
ensemble_eval.py
================
Phase 3 evaluation: WBF ensemble (YOLOv11 + YOLOv26) vs COCO GT annotations.

For each condition, runs both models at conf=0.07, applies WBF fusion + adaptive
post-processing, then evaluates against ground truth using the same mAP50 /
precision / recall / F1 protocol as val_condition_eval.py.

Results written to:
  /nfsshare/joseva/condition_eval_results/ensemble_<condition>.json

Author: Toni (Claude Sonnet 4.6), 2026-04-04
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
CONDITION_SPLIT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")
VAL_IMG_DIR = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val")
COCO_JSON = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_val.json")
RESULTS_DIR = Path("/nfsshare/joseva/condition_eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "yolo11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
}

GPU = 0
PRE_CONF = 0.07
MODEL_WEIGHTS = [0.55, 0.45]  # YOLOv11, YOLOv26

WBF_IOU = {0: 0.45, 1: 0.50, 2: 0.55}   # helmet, vest, worker
POST_WBF_GLOBAL = {0: 0.20, 1: 0.22, 2: 0.18}

CLS_NAMES = ["helmet", "safety_vest", "worker"]
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]

sys.path.insert(0, "/nfsshare/joseva")
from adaptive_postprocess import apply_all_rules


# ── WBF (same as ensemble_batch.py) ───────────────────────────────────────────

def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)


def wbf_fuse(all_preds, img_w, img_h):
    flat = []
    for model_idx, (boxes, scores, labels) in enumerate(all_preds):
        for box, score, label in zip(boxes, scores, labels):
            norm = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
            flat.append((norm, score, label, model_idx))

    if not flat:
        return []

    result = []
    for cls_id in (0, 1, 2):
        cls_flat = [(b, s, mi) for b, s, l, mi in flat if l == cls_id]
        if not cls_flat:
            continue
        iou_thresh = WBF_IOU[cls_id]
        cls_flat.sort(key=lambda x: -x[1])
        used = [False] * len(cls_flat)
        for i in range(len(cls_flat)):
            if used[i]:
                continue
            cluster_boxes = [cls_flat[i][0]]
            cluster_scores = [cls_flat[i][1]]
            cluster_midxs = [cls_flat[i][2]]
            used[i] = True
            for j in range(i + 1, len(cls_flat)):
                if used[j]:
                    continue
                if iou_box(cls_flat[i][0], cls_flat[j][0]) >= iou_thresh:
                    cluster_boxes.append(cls_flat[j][0])
                    cluster_scores.append(cls_flat[j][1])
                    cluster_midxs.append(cls_flat[j][2])
                    used[j] = True
            raw_w = np.array([MODEL_WEIGHTS[mi] * s for mi, s in zip(cluster_midxs, cluster_scores)])
            norm_w = raw_w / raw_w.sum()
            fused_score = float(np.average(cluster_scores, weights=norm_w))
            if fused_score < POST_WBF_GLOBAL[cls_id]:
                continue
            fused_box_norm = np.average(cluster_boxes, axis=0, weights=norm_w)
            fused_box = [
                fused_box_norm[0] * img_w, fused_box_norm[1] * img_h,
                fused_box_norm[2] * img_w, fused_box_norm[3] * img_h,
            ]
            result.append({"box": fused_box, "score": fused_score, "cls": cls_id})
    return result


# ── metric computation (same protocol as val_condition_eval.py) ────────────────

def compute_ap(recalls, precisions):
    ap = 0.0
    for thr in [i / 10 for i in range(11)]:
        p_at_r = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def match_predictions(preds, gts, iou_thr=0.50):
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


def compute_metrics(all_results, all_fn):
    per_class = {}
    for c in range(3):
        cls_res = sorted([r for r in all_results if r["cls"] == c], key=lambda x: -x["score"])
        fn = sum(d.get(c, 0) for d in all_fn)
        tp_c, fp_c, recalls, precisions = 0, 0, [], []
        total_gt = sum(r["tp"] for r in cls_res) + fn
        for r in cls_res:
            tp_c += r["tp"]; fp_c += r["fp"]
            recalls.append(tp_c / max(total_gt, 1))
            precisions.append(tp_c / max(tp_c + fp_c, 1))
        ap = compute_ap(recalls, precisions) if recalls else 0.0
        rec = recalls[-1] if recalls else 0.0
        pre = precisions[-1] if precisions else 0.0
        f1 = 2 * pre * rec / max(pre + rec, 1e-6)
        per_class[CLS_NAMES[c]] = {
            "AP50": round(ap, 4), "precision": round(pre, 4),
            "recall": round(rec, 4), "F1": round(f1, 4),
            "TP": tp_c, "FP": fp_c, "FN": fn,
        }
    aps = [v["AP50"] for v in per_class.values()]
    pres = [v["precision"] for v in per_class.values()]
    recs = [v["recall"] for v in per_class.values()]
    mP = round(sum(pres) / 3, 4)
    mR = round(sum(recs) / 3, 4)
    return {
        "mAP50":     round(sum(aps) / 3, 4),
        "precision": mP,
        "recall":    mR,
        "F1":        round(2 * mP * mR / max(mP + mR, 1e-6), 4),
        "total_FP":  sum(v["FP"] for v in per_class.values()),
        "total_FN":  sum(v["FN"] for v in per_class.values()),
        "per_class": per_class,
    }


# ── main eval loop ─────────────────────────────────────────────────────────────

def run():
    from ultralytics import YOLO

    print("Loading models...")
    model_v11 = YOLO(MODEL_PATHS["yolo11"])
    model_v26 = YOLO(MODEL_PATHS["yolo26"])
    print("Models loaded.\n")

    coco_data = json.loads(COCO_JSON.read_text())
    splits = json.loads(CONDITION_SPLIT_JSON.read_text())

    id2fname = {img["id"]: Path(img["file_name"]).name for img in coco_data["images"]}
    gt_by_file = {Path(img["file_name"]).name: [] for img in coco_data["images"]}
    for ann in coco_data["annotations"]:
        fname = id2fname.get(ann["image_id"])
        if fname and fname in gt_by_file:
            x, y, w, h = ann["bbox"]
            gt_by_file[fname].append({"box": [x, y, x+w, y+h], "cls": ann["category_id"] - 1})

    all_condition_metrics = {}

    for condition in CONDITIONS:
        file_names = splits.get(condition, [])
        if not file_names:
            print(f"  {condition}: no images, skipping")
            continue

        print(f"=== {condition}: {len(file_names)} images ===")
        all_results, all_fn = [], []
        total_time, n_imgs = 0.0, 0

        for fname in file_names:
            img_path = VAL_IMG_DIR / fname
            if not img_path.exists() or fname not in gt_by_file:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            gts = gt_by_file[fname]

            t0 = time.perf_counter()

            # Run both models
            all_preds = []
            for model in (model_v11, model_v26):
                r = model.predict(str(img_path), device=GPU, verbose=False,
                                  conf=PRE_CONF, iou=0.35)[0]
                boxes, scores, labels = [], [], []
                for box in (r.boxes or []):
                    xy = box.xyxy[0].cpu().numpy().tolist()
                    boxes.append(xy)
                    scores.append(float(box.conf[0]))
                    labels.append(int(box.cls[0]))
                all_preds.append((boxes, scores, labels))

            # WBF fusion
            fused = wbf_fuse(all_preds, img_w, img_h)

            # Adaptive post-processing
            final_boxes, _ = apply_all_rules(fused, condition, img_w, img_h)

            total_time += time.perf_counter() - t0
            n_imgs += 1

            # Match against GT
            preds = [{"box": b["box"], "score": b["score"], "cls": b["cls"]}
                     for b in final_boxes]
            res, _ = match_predictions(preds, gts)
            all_results.extend(res)

            fn_by_cls = {}
            matched_cls = [r["cls"] for r in res if r["tp"] == 1]
            for gt in gts:
                c = gt["cls"]
                fn_by_cls[c] = fn_by_cls.get(c, 0) + 1
            for c in matched_cls:
                fn_by_cls[c] = max(0, fn_by_cls.get(c, 0) - 1)
            all_fn.append(fn_by_cls)

        metrics = compute_metrics(all_results, all_fn)
        fps = round(n_imgs / total_time, 2) if total_time > 0 else 0.0
        metrics["FPS"] = fps
        metrics["n_imgs"] = n_imgs

        out_path = RESULTS_DIR / f"ensemble_{condition}.json"
        with open(out_path, "w") as f:
            json.dump({"model": "Ensemble_WBF", "condition": condition, **metrics}, f, indent=2)

        all_condition_metrics[condition] = metrics
        print(f"  mAP50={metrics['mAP50']}  F1={metrics['F1']}  "
              f"P={metrics['precision']}  R={metrics['recall']}  FPS={fps}")
        print(f"  helmet:  AP={metrics['per_class']['helmet']['AP50']}  "
              f"R={metrics['per_class']['helmet']['recall']}")
        print(f"  vest:    AP={metrics['per_class']['safety_vest']['AP50']}  "
              f"R={metrics['per_class']['safety_vest']['recall']}")
        print(f"  worker:  AP={metrics['per_class']['worker']['AP50']}  "
              f"R={metrics['per_class']['worker']['recall']}")
        print(f"  Saved -> {out_path}\n")

    # ── Overall summary ─────────────────────────────────────────────────────────
    if all_condition_metrics:
        all_maps = [m["mAP50"] for m in all_condition_metrics.values()]
        all_f1s  = [m["F1"]    for m in all_condition_metrics.values()]
        print("=" * 60)
        print("ENSEMBLE EVAL COMPLETE")
        print(f"  Mean mAP50 across conditions : {round(sum(all_maps)/len(all_maps), 4)}")
        print(f"  Mean F1    across conditions : {round(sum(all_f1s)/len(all_f1s), 4)}")
        print("=" * 60)

        # ntfy
        import subprocess
        mean_map = round(sum(all_maps)/len(all_maps), 4)
        mean_f1  = round(sum(all_f1s)/len(all_f1s), 4)
        vest_recalls = [m["per_class"]["safety_vest"]["recall"]
                        for m in all_condition_metrics.values()]
        mean_vest_r = round(sum(vest_recalls)/len(vest_recalls), 4)
        subprocess.run([
            "curl", "-s", "-X", "POST",
            "-H", "Title: TONI: Ensemble Eval Complete",
            "-H", "Priority: high",
            "-d", (
                f"Phase 3 eval done. "
                f"Ensemble mAP50={mean_map} F1={mean_f1} "
                f"vest_recall={mean_vest_r}. "
                f"Results: condition_eval_results/ensemble_*.json"
            ),
            "ntfy.sh/buildsight-tournament-2026"
        ], check=False)


if __name__ == "__main__":
    run()
