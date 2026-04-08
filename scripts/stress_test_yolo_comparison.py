"""
BuildSight — YOLOv11 vs YOLOv26 Structured Stress Test
=========================================================
Evaluates both models across 4 site conditions:
  S1: Normal       — clean val set, full YOLO .val() metrics
  S2: Dusty        — haze + blur augmentation
  S3: Low-Light    — brightness 35% reduction
  S4: Crowded      — images with ≥5 annotated objects

Metrics per condition:
  - Precision, Recall, mAP50, mAP50-95
  - Per-class AP50 (helmet, safety_vest, worker)
  - False Positives, False Negatives
  - Detection stability score
  - Avg confidence, Avg latency (ms)

Outputs:
  ~/stress_report.txt     — full text report
  ~/stress_results.json   — machine-readable JSON
  ~/stress_samples/       — annotated sample images (3 per condition per model)
"""

import cv2
import glob
import json
import numpy as np
import os
import re
import time
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)"
V11_DIR  = f"{BASE}/YOLOv11"
V26_DIR  = f"{BASE}/YOLOv26"
V11_BEST = f"{V11_DIR}/runs/detect/runs/train/buildsight/weights/best.pt"
V26_BEST = f"{V26_DIR}/runs/detect/runs/train/buildsight/weights/best.pt"
VAL_IMGS = f"{V11_DIR}/images/val"
VAL_LBLS = f"{V11_DIR}/labels/val"
V11_YAML = f"{V11_DIR}/data.yaml"
V26_YAML = f"{V26_DIR}/data.yaml"

HOME     = os.path.expanduser("~")
REPORT   = f"{HOME}/stress_report.txt"
JSON_OUT = f"{HOME}/stress_results.json"
SAMPLES  = f"{HOME}/stress_samples"

GPU      = 0           # A100 = CUDA device 0
MAX_IMGS = 300         # cap per condition (speed vs coverage)
CROWDED_MIN = 5        # min objects to count as crowded scene
CONF_THR    = 0.25
IOU_THR     = 0.5      # for TP/FP/FN matching

CLASS_NAMES = ["helmet", "safety_vest", "worker"]
CLS_COLORS  = [(0, 220, 0), (0, 165, 255), (0, 0, 255)]  # green, orange, red

os.makedirs(SAMPLES, exist_ok=True)
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)

def sep(title="", width=68):
    bar = "=" * width
    log(bar)
    if title:
        log(f"  {title}")
        log(bar)

def subsep(title):
    log(f"\n  {'─'*60}")
    log(f"  {title}")
    log(f"  {'─'*60}")


# ── Augmentations ──────────────────────────────────────────────────────────────
def augment_dusty(img):
    """Simulate dusty/hazy site: whitish overlay + Gaussian blur."""
    haze = np.full_like(img, 185, dtype=np.uint8)
    out  = cv2.addWeighted(img, 0.58, haze, 0.42, 0)
    return cv2.GaussianBlur(out, (5, 5), 1.2)

def augment_lowlight(img):
    """Simulate night / poor illumination: darken to 35% brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= 0.35
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def augment_none(img):
    return img


# ── Label parsing ──────────────────────────────────────────────────────────────
def parse_label(lbl_path, img_w, img_h):
    """Returns list of (cls, x1,y1,x2,y2) in pixel coords."""
    boxes = []
    if not os.path.exists(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw/2) * img_w)
            y1 = int((cy - bh/2) * img_h)
            x2 = int((cx + bw/2) * img_w)
            y2 = int((cy + bh/2) * img_h)
            boxes.append((cls, x1, y1, x2, y2))
    return boxes

def iou_box(a, b):
    """IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = (a[2]-a[0]) * (a[3]-a[1])
    ab = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(aa + ab - inter, 1e-6)


# ── Image list helpers ─────────────────────────────────────────────────────────
def get_all_val_images():
    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(glob.glob(os.path.join(VAL_IMGS, ext)))
    return sorted(imgs)

def get_crowded_images(all_imgs, min_obj=CROWDED_MIN):
    crowded = []
    for img_path in all_imgs:
        name = Path(img_path).stem
        lbl  = os.path.join(VAL_LBLS, name + ".txt")
        if not os.path.exists(lbl):
            continue
        with open(lbl) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) >= min_obj:
            crowded.append(img_path)
    return crowded


# ── Core evaluation ────────────────────────────────────────────────────────────
def evaluate_condition(model, images, aug_fn, condition_name, model_name,
                       max_imgs=MAX_IMGS, save_samples=True):
    """
    Run inference on augmented images, compute metrics vs ground-truth labels.
    Returns dict of metrics.
    """
    imgs = images[:max_imgs]
    total  = len(imgs)

    # Accumulators
    all_confs, latencies = [], []
    tp_total = fp_total = fn_total = 0
    per_cls_tp   = [0, 0, 0]
    per_cls_fp   = [0, 0, 0]
    per_cls_fn   = [0, 0, 0]
    det_counts   = []   # detections per image (for stability)
    sample_paths = []

    for idx, img_path in enumerate(imgs):
        img_orig = cv2.imread(img_path)
        if img_orig is None:
            continue

        h, w = img_orig.shape[:2]
        img_aug = aug_fn(img_orig)

        # Inference
        t0 = time.time()
        result = model.predict(img_aug, device=GPU, verbose=False, conf=CONF_THR)[0]
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        # Predicted boxes
        pred_boxes = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xy     = box.xyxy[0].cpu().numpy()
                pred_boxes.append((cls_id, conf, int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])))
                all_confs.append(conf)

        det_counts.append(len(pred_boxes))

        # Ground-truth boxes
        name    = Path(img_path).stem
        lbl_path = os.path.join(VAL_LBLS, name + ".txt")
        gt_boxes = parse_label(lbl_path, w, h)

        # TP / FP / FN matching (greedy, per class)
        matched_gt  = set()
        matched_pred = set()
        for pi, (pcls, pconf, px1, py1, px2, py2) in enumerate(pred_boxes):
            best_iou, best_gi = 0, -1
            for gi, (gcls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
                if gi in matched_gt or gcls != pcls:
                    continue
                iou = iou_box((px1,py1,px2,py2), (gx1,gy1,gx2,gy2))
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= IOU_THR and best_gi >= 0:
                tp_total += 1
                per_cls_tp[pcls] += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                fp_total += 1
                per_cls_fp[pcls] += 1

        for gi, (gcls, *_) in enumerate(gt_boxes):
            if gi not in matched_gt:
                fn_total += 1
                per_cls_fn[gcls] += 1

        # Save annotated samples (first 3 per condition)
        if save_samples and idx < 3:
            vis = img_aug.copy()
            # Draw GT (dashed blue)
            for gcls, gx1, gy1, gx2, gy2 in gt_boxes:
                for dash in range(gx1, gx2, 10):
                    cv2.line(vis, (dash, gy1), (min(dash+6, gx2), gy1), (255,100,0), 1)
                    cv2.line(vis, (dash, gy2), (min(dash+6, gx2), gy2), (255,100,0), 1)
                for dash in range(gy1, gy2, 10):
                    cv2.line(vis, (gx1, dash), (gx1, min(dash+6, gy2)), (255,100,0), 1)
                    cv2.line(vis, (gx2, dash), (gx2, min(dash+6, gy2)), (255,100,0), 1)
            # Draw predictions
            for pcls, pconf, px1, py1, px2, py2 in pred_boxes:
                color = CLS_COLORS[pcls % 3]
                cv2.rectangle(vis, (px1,py1), (px2,py2), color, 2)
                label = f"{CLASS_NAMES[pcls]} {pconf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (px1, py1-th-4), (px1+tw+2, py1), color, -1)
                cv2.putText(vis, label, (px1+1, py1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            # Watermark
            tag = f"{model_name} | {condition_name} | img {idx+1}"
            cv2.putText(vis, tag, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            fname = f"{SAMPLES}/{model_name}_{condition_name}_sample{idx+1}.jpg"
            cv2.imwrite(fname, vis)
            sample_paths.append(fname)

    # Metrics calculation
    n = len(imgs)
    precision = tp_total / max(tp_total + fp_total, 1)
    recall    = tp_total / max(tp_total + fn_total, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-6)
    avg_conf  = float(np.mean(all_confs)) if all_confs else 0.0
    avg_lat   = float(np.mean(latencies)) if latencies else 0.0

    # Detection stability: 1 - (std/mean) of detection counts
    det_arr = np.array(det_counts, dtype=float)
    stability = round(1.0 - (np.std(det_arr) / max(np.mean(det_arr), 1.0)), 4)
    stability = max(0.0, stability)

    # Per-class precision / recall
    per_class = {}
    for i, cname in enumerate(CLASS_NAMES):
        pc_prec = per_cls_tp[i] / max(per_cls_tp[i] + per_cls_fp[i], 1)
        pc_rec  = per_cls_tp[i] / max(per_cls_tp[i] + per_cls_fn[i], 1)
        per_class[cname] = {
            "precision": round(pc_prec, 4),
            "recall":    round(pc_rec, 4),
            "tp": per_cls_tp[i], "fp": per_cls_fp[i], "fn": per_cls_fn[i],
        }

    return {
        "images_evaluated": n,
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
        "fp_total":   fp_total,
        "fn_total":   fn_total,
        "tp_total":   tp_total,
        "avg_conf":   round(avg_conf, 4),
        "avg_lat_ms": round(avg_lat, 2),
        "stability":  stability,
        "per_class":  per_class,
        "samples":    sample_paths,
    }


# ── S1: Full YOLO .val() for mAP50 / mAP50-95 ─────────────────────────────────
def run_official_val(model, yaml_path, model_name):
    """Run official YOLO validation for mAP numbers."""
    log(f"    Running official .val() for {model_name}...")
    try:
        r = model.val(data=yaml_path, device=GPU, verbose=False, conf=CONF_THR)
        map50    = round(float(r.box.map50), 4)
        map5095  = round(float(r.box.map),   4)
        prec     = round(float(r.box.mp),    4)
        rec      = round(float(r.box.mr),    4)
        f1_val   = round(2*prec*rec / max(prec+rec, 1e-6), 4)
        inf_ms   = round(r.speed.get("inference", 0), 2) if hasattr(r, "speed") else 0
        per_cls  = {}
        if hasattr(r.box, "ap50") and r.box.ap50 is not None:
            for i, cn in model.names.items():
                if i < len(r.box.ap50):
                    per_cls[cn] = round(float(r.box.ap50[i]), 4)
        return {
            "mAP50": map50, "mAP50_95": map5095,
            "precision": prec, "recall": rec, "f1": f1_val,
            "inference_ms": inf_ms, "per_class_ap50": per_cls,
        }
    except Exception as e:
        log(f"    [WARN] .val() failed: {e}")
        return {"mAP50": 0, "mAP50_95": 0, "precision": 0, "recall": 0,
                "f1": 0, "inference_ms": 0, "per_class_ap50": {}}


# ── Print condition block ──────────────────────────────────────────────────────
def print_condition(cname, m11, m26):
    subsep(f"Condition: {cname}")
    hdr = f"  {'Metric':<26} {'YOLOv11':>12} {'YOLOv26':>12}  {'Winner':>10}"
    log(hdr)
    log(f"  {'─'*60}")

    def row(label, k11, k26, fmt=".4f", higher_better=True):
        v11 = m11.get(k11, 0) if isinstance(k11, str) else k11
        v26 = m26.get(k26, 0) if isinstance(k26, str) else k26
        if isinstance(v11, float) and isinstance(v26, float):
            if higher_better:
                win = "YOLOv11" if v11 > v26 else ("YOLOv26" if v26 > v11 else "Tie")
            else:
                win = "YOLOv11" if v11 < v26 else ("YOLOv26" if v26 < v11 else "Tie")
            log(f"  {label:<26} {v11:>12{fmt}} {v26:>12{fmt}}  {win:>10}")
        else:
            log(f"  {label:<26} {str(v11):>12} {str(v26):>12}")

    row("Precision",        "precision",  "precision")
    row("Recall",           "recall",     "recall")
    row("F1 Score",         "f1",         "f1")
    row("TP (true pos)",    "tp_total",   "tp_total",   "d")
    row("FP (false pos)",   "fp_total",   "fp_total",   "d", higher_better=False)
    row("FN (false neg)",   "fn_total",   "fn_total",   "d", higher_better=False)
    row("Avg Confidence",   "avg_conf",   "avg_conf")
    row("Avg Latency (ms)", "avg_lat_ms", "avg_lat_ms", ".2f", higher_better=False)
    row("Det. Stability",   "stability",  "stability")

    if "mAP50" in m11:
        row("mAP50",        "mAP50",      "mAP50")
        row("mAP50-95",     "mAP50_95",   "mAP50_95")
        pc11 = m11.get("per_class_ap50", {})
        pc26 = m26.get("per_class_ap50", {})
        if pc11 or pc26:
            log(f"\n  Per-Class AP50:")
            for cn in CLASS_NAMES:
                v11 = pc11.get(cn, 0.0)
                v26 = pc26.get(cn, 0.0)
                win = "YOLOv11" if v11 > v26 else ("YOLOv26" if v26 > v11 else "Tie")
                log(f"    {cn:<20} {v11:>10.4f} {v26:>10.4f}  {win:>10}")

    # Per-class breakdown
    pc11 = m11.get("per_class", {})
    pc26 = m26.get("per_class", {})
    if pc11 and pc26:
        log(f"\n  Per-Class Precision / Recall:")
        log(f"  {'Class':<16} {'V11 Prec':>10} {'V11 Rec':>9} {'V26 Prec':>10} {'V26 Rec':>9}")
        log(f"  {'─'*56}")
        for cn in CLASS_NAMES:
            p11 = pc11.get(cn, {}); p26 = pc26.get(cn, {})
            log(f"  {cn:<16} {p11.get('precision',0):>10.4f} {p11.get('recall',0):>9.4f} "
                f"{p26.get('precision',0):>10.4f} {p26.get('recall',0):>9.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from ultralytics import YOLO

    start_time = datetime.now()
    sep(f"BuildSight — YOLOv11 vs YOLOv26 Stress Test  [{start_time:%Y-%m-%d %H:%M}]")
    log(f"  Models:     YOLOv11 → {V11_BEST}")
    log(f"              YOLOv26 → {V26_BEST}")
    log(f"  Val images: {VAL_IMGS}")
    log(f"  GPU:        CUDA:{GPU} (A100)")
    log(f"  Conf thr:   {CONF_THR}  |  IoU thr (TP match): {IOU_THR}")
    log(f"  Max imgs/cond: {MAX_IMGS}")

    # Load models
    log("\n  Loading models...")
    m11 = YOLO(V11_BEST)
    m26 = YOLO(V26_BEST)
    log("  Both models loaded.")

    # Image lists
    all_imgs    = get_all_val_images()
    crowded_imgs = get_crowded_images(all_imgs)
    log(f"\n  Total val images : {len(all_imgs)}")
    log(f"  Crowded (≥{CROWDED_MIN} obj): {len(crowded_imgs)}")

    results = {
        "YOLOv11": {"model": "YOLOv11", "weights": V11_BEST},
        "YOLOv26": {"model": "YOLOv26", "weights": V26_BEST},
    }

    # ──────────────────────────────────────────────────────────────────────────
    #  S1: NORMAL — official .val() + inference metrics
    # ──────────────────────────────────────────────────────────────────────────
    sep("[S1] NORMAL — Full Validation Metrics")

    log("  [YOLOv11]")
    s1_map_v11 = run_official_val(m11, V11_YAML, "YOLOv11")
    log("  [YOLOv26]")
    s1_map_v26 = run_official_val(m26, V26_YAML, "YOLOv26")

    log("  Running inference-level metrics (TP/FP/FN)...")
    s1_v11 = evaluate_condition(m11, all_imgs, augment_none, "Normal", "YOLOv11")
    s1_v26 = evaluate_condition(m26, all_imgs, augment_none, "Normal", "YOLOv26")

    # Merge mAP into s1 dicts
    s1_v11.update(s1_map_v11)
    s1_v26.update(s1_map_v26)
    results["YOLOv11"]["S1"] = s1_v11
    results["YOLOv26"]["S1"] = s1_v26
    print_condition("S1 — Normal", s1_v11, s1_v26)

    # ──────────────────────────────────────────────────────────────────────────
    #  S2: DUSTY
    # ──────────────────────────────────────────────────────────────────────────
    sep("[S2] DUSTY — Haze + Blur Augmentation")
    log("  [YOLOv11]")
    s2_v11 = evaluate_condition(m11, all_imgs, augment_dusty, "Dusty", "YOLOv11")
    log("  [YOLOv26]")
    s2_v26 = evaluate_condition(m26, all_imgs, augment_dusty, "Dusty", "YOLOv26")
    results["YOLOv11"]["S2"] = s2_v11
    results["YOLOv26"]["S2"] = s2_v26
    print_condition("S2 — Dusty", s2_v11, s2_v26)

    # ──────────────────────────────────────────────────────────────────────────
    #  S3: LOW-LIGHT
    # ──────────────────────────────────────────────────────────────────────────
    sep("[S3] LOW-LIGHT — 35% Brightness")
    log("  [YOLOv11]")
    s3_v11 = evaluate_condition(m11, all_imgs, augment_lowlight, "LowLight", "YOLOv11")
    log("  [YOLOv26]")
    s3_v26 = evaluate_condition(m26, all_imgs, augment_lowlight, "LowLight", "YOLOv26")
    results["YOLOv11"]["S3"] = s3_v11
    results["YOLOv26"]["S3"] = s3_v26
    print_condition("S3 — Low-Light", s3_v11, s3_v26)

    # ──────────────────────────────────────────────────────────────────────────
    #  S4: CROWDED
    # ──────────────────────────────────────────────────────────────────────────
    sep(f"[S4] CROWDED — Scenes with ≥{CROWDED_MIN} Objects")
    log(f"  Using {min(len(crowded_imgs), MAX_IMGS)} crowded images")
    eval_set = crowded_imgs if crowded_imgs else all_imgs[:50]
    log("  [YOLOv11]")
    s4_v11 = evaluate_condition(m11, eval_set, augment_none, "Crowded", "YOLOv11")
    log("  [YOLOv26]")
    s4_v26 = evaluate_condition(m26, eval_set, augment_none, "Crowded", "YOLOv26")
    results["YOLOv11"]["S4"] = s4_v11
    results["YOLOv26"]["S4"] = s4_v26
    print_condition("S4 — Crowded", s4_v11, s4_v26)

    # ──────────────────────────────────────────────────────────────────────────
    #  OVERALL SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    sep("OVERALL SUMMARY — ALL CONDITIONS")

    conds     = ["S1", "S2", "S3", "S4"]
    cond_names= ["Normal", "Dusty", "Low-Light", "Crowded"]
    metrics   = ["precision", "recall", "f1", "avg_conf", "stability"]

    # Win tally
    wins = {"YOLOv11": 0, "YOLOv26": 0}
    total_comparisons = 0

    log(f"\n  {'Metric':<20} {'Cond':<12} {'YOLOv11':>10} {'YOLOv26':>10}  {'Winner':>10}")
    log(f"  {'─'*64}")
    for cond, cname in zip(conds, cond_names):
        for metric in metrics:
            v11_val = results["YOLOv11"].get(cond, {}).get(metric, 0)
            v26_val = results["YOLOv26"].get(cond, {}).get(metric, 0)
            if isinstance(v11_val, float) and isinstance(v26_val, float):
                if v11_val > v26_val:
                    win = "YOLOv11"; wins["YOLOv11"] += 1
                elif v26_val > v11_val:
                    win = "YOLOv26"; wins["YOLOv26"] += 1
                else:
                    win = "Tie"
                total_comparisons += 1
                log(f"  {metric:<20} {cname:<12} {v11_val:>10.4f} {v26_val:>10.4f}  {win:>10}")

    # mAP50 comparison (S1 only — official)
    log(f"\n  {'mAP50 (official)':<20} {'Normal':<12} "
        f"{s1_v11.get('mAP50',0):>10.4f} {s1_v26.get('mAP50',0):>10.4f}  "
        f"{'YOLOv11' if s1_v11.get('mAP50',0)>s1_v26.get('mAP50',0) else 'YOLOv26':>10}")
    log(f"  {'mAP50-95 (official)':<20} {'Normal':<12} "
        f"{s1_v11.get('mAP50_95',0):>10.4f} {s1_v26.get('mAP50_95',0):>10.4f}  "
        f"{'YOLOv11' if s1_v11.get('mAP50_95',0)>s1_v26.get('mAP50_95',0) else 'YOLOv26':>10}")

    # FP / FN across conditions
    log(f"\n  False Positive / False Negative Count by Condition:")
    log(f"  {'Condition':<14} {'V11 FP':>8} {'V11 FN':>8} {'V26 FP':>8} {'V26 FN':>8}")
    log(f"  {'─'*48}")
    for cond, cname in zip(conds, cond_names):
        v11r = results["YOLOv11"].get(cond, {})
        v26r = results["YOLOv26"].get(cond, {})
        log(f"  {cname:<14} {v11r.get('fp_total',0):>8d} {v11r.get('fn_total',0):>8d} "
            f"{v26r.get('fp_total',0):>8d} {v26r.get('fn_total',0):>8d}")

    # Latency table
    log(f"\n  Inference Latency — Avg ms/image:")
    log(f"  {'Condition':<14} {'YOLOv11 ms':>12} {'YOLOv26 ms':>12}  {'Faster':>10}")
    log(f"  {'─'*50}")
    for cond, cname in zip(conds, cond_names):
        v11_lat = results["YOLOv11"].get(cond, {}).get("avg_lat_ms", 0)
        v26_lat = results["YOLOv26"].get(cond, {}).get("avg_lat_ms", 0)
        faster = "YOLOv11" if v11_lat < v26_lat else ("YOLOv26" if v26_lat < v11_lat else "Tie")
        log(f"  {cname:<14} {v11_lat:>12.2f} {v26_lat:>12.2f}  {faster:>10}")

    # Final verdict
    sep("VERDICT")
    log(f"\n  Win count (higher = better) across {total_comparisons} metric comparisons:")
    log(f"  YOLOv11:  {wins['YOLOv11']} wins")
    log(f"  YOLOv26:  {wins['YOLOv26']} wins")

    overall_winner = "YOLOv11" if wins["YOLOv11"] >= wins["YOLOv26"] else "YOLOv26"
    runner_up      = "YOLOv26" if overall_winner == "YOLOv11" else "YOLOv11"

    log(f"\n  ★  OVERALL STRESS TEST WINNER: {overall_winner}")
    log(f"     Runner-up:                   {runner_up}")
    log(f"\n  Note: Both models will be re-evaluated in the full tournament")
    log(f"  benchmark (weekend_benchmark.py) after YOLACT++ training completes.")

    elapsed = datetime.now() - start_time
    log(f"\n  Total time: {elapsed}")
    log(f"  Samples saved to: {SAMPLES}/")
    sep()

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(REPORT, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved → {REPORT}")

    # Clean results for JSON (remove non-serializable items)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k != "samples"}
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    with open(JSON_OUT, "w") as f:
        json.dump(clean(results), f, indent=2)
    print(f"JSON saved  → {JSON_OUT}")
    print(f"Samples     → {SAMPLES}/")
