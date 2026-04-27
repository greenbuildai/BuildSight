#!/usr/bin/env python3
"""
ensemble_batch.py
=================
BuildSight Phase 3 — Multi-Model Ensemble (WBF + Adaptive Post-Processing)

Pipeline per image:
  1. YOLOv11 inference at conf=0.07
  2. YOLOv26 inference at conf=0.07
  3. Per-class Weighted Box Fusion (worker IoU=0.55, vest IoU=0.50, helmet IoU=0.45)
     Weights: YOLOv11=0.55, YOLOv26=0.45
  4. apply_all_rules() from adaptive_postprocess.py (per-condition 8-rule system)
  5. Save annotated image + CSV summary

Output: /nfsshare/joseva/val_annotated_ensemble/<condition>/Ensemble/

Author: Toni (Claude Sonnet 4.6), 2026-04-04
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
CONDITION_SPLIT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")
VAL_IMG_DIR = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val")
OUT_DIR = Path("/nfsshare/joseva/val_annotated_ensemble")
LOG_DIR = Path("/nfsshare/joseva/logs")

MODEL_PATHS = {
    "yolo11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
}

# ── WBF parameters (from ensemble_strategy.md) ─────────────────────────────────
GPU = 0
PRE_CONF = 0.07          # loose gate — let both models contribute weak evidence
MODEL_WEIGHTS = [0.55, 0.45]   # YOLOv11=0.55, YOLOv26=0.45

# Early-exit: skip YOLOv26 when YOLOv11 is uniformly high-confidence
# If ALL v11 detections >= this threshold → treat v11 result as final, skip v26
EARLY_EXIT_CONF = 0.65
EARLY_EXIT_MIN_DETS = 1   # must have at least this many detections to early-exit

# Per-class WBF IoU thresholds
WBF_IOU = {
    0: 0.45,   # helmet — small, position varies; looser merge
    1: 0.50,   # safety_vest — pose-dependent box shape
    2: 0.55,   # worker — clear body boundaries; tighter consensus
}

# Post-WBF global confidence gate (applied before adaptive per-condition thresholds)
POST_WBF_GLOBAL = {
    0: 0.20,   # helmet
    1: 0.22,   # safety_vest
    2: 0.18,   # worker
}

CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]
CLS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}
CLS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 100, 0)}


# ── WBF implementation ─────────────────────────────────────────────────────────

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)


def wbf_fuse(all_preds, img_w, img_h):
    """
    all_preds: list of (boxes_xyxy, scores, labels) — one entry per model
    Returns list of dicts: {"box": [x1,y1,x2,y2], "score": float, "cls": int}
    """
    # Flatten with model-weight tracking
    flat = []  # (box_norm, score, label, model_idx)
    for model_idx, (boxes, scores, labels) in enumerate(all_preds):
        for box, score, label in zip(boxes, scores, labels):
            # Normalise to [0,1]
            norm = [
                box[0] / img_w, box[1] / img_h,
                box[2] / img_w, box[3] / img_h,
            ]
            flat.append((norm, score, label, model_idx))

    if not flat:
        return []

    # Process per class
    result = []
    for cls_id in (0, 1, 2):
        cls_flat = [(b, s, mi) for b, s, l, mi in flat if l == cls_id]
        if not cls_flat:
            continue

        iou_thresh = WBF_IOU[cls_id]

        # Sort by score descending
        cls_flat.sort(key=lambda x: -x[1])

        used = [False] * len(cls_flat)
        for i in range(len(cls_flat)):
            if used[i]:
                continue

            cluster_boxes = [cls_flat[i][0]]
            cluster_scores = [cls_flat[i][1]]
            cluster_model_idxs = [cls_flat[i][2]]
            used[i] = True

            for j in range(i + 1, len(cls_flat)):
                if used[j]:
                    continue
                if iou(cls_flat[i][0], cls_flat[j][0]) >= iou_thresh:
                    cluster_boxes.append(cls_flat[j][0])
                    cluster_scores.append(cls_flat[j][1])
                    cluster_model_idxs.append(cls_flat[j][2])
                    used[j] = True

            # Weighted average of box coordinates
            # Weight each box by its model's global weight × confidence
            raw_w = np.array([
                MODEL_WEIGHTS[mi] * s
                for mi, s in zip(cluster_model_idxs, cluster_scores)
            ])
            norm_w = raw_w / raw_w.sum()
            fused_box_norm = np.average(cluster_boxes, axis=0, weights=norm_w)

            # Fused score = weighted mean of confidences (model-weight-scaled)
            fused_score = float(np.average(cluster_scores, weights=norm_w))

            # Global gate
            if fused_score < POST_WBF_GLOBAL[cls_id]:
                continue

            # Denormalise
            fused_box = [
                fused_box_norm[0] * img_w,
                fused_box_norm[1] * img_h,
                fused_box_norm[2] * img_w,
                fused_box_norm[3] * img_h,
            ]

            result.append({"box": fused_box, "score": fused_score, "cls": cls_id})

    return result


# ── drawing ─────────────────────────────────────────────────────────────────────

def draw_boxes(image, boxes, condition, stats):
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["box"]]
        color = CLS_COLORS[box["cls"]]
        label = f"{CLS_NAMES[box['cls']]} {box['score']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(image, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    raw = stats.get("raw", "?")
    final = stats.get("final", len(boxes))
    banner = (
        f"[ENSEMBLE-WBF] {condition} "
        f"raw:{raw}->final:{final} (-{raw - final if isinstance(raw, int) else '?'})"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(image, banner, (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    return image


# ── adaptive post-processing (imported from adaptive_postprocess.py) ────────────

sys.path.insert(0, str(Path(__file__).parent.parent))  # local dev
sys.path.insert(0, "/nfsshare/joseva")                 # SASTRA node1
from adaptive_postprocess import apply_all_rules


# ── main batch loop ─────────────────────────────────────────────────────────────

def run():
    from ultralytics import YOLO

    splits = json.loads(CONDITION_SPLIT_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    model_v11 = YOLO(MODEL_PATHS["yolo11"])
    model_v26 = YOLO(MODEL_PATHS["yolo26"])
    print("Models loaded.")

    csv_rows = []
    total_images = 0
    total_raw = 0
    total_final = 0

    for condition in CONDITIONS:
        file_names = splits.get(condition, [])
        if not file_names:
            print(f"  {condition}: no images in split, skipping")
            continue

        dst_dir = OUT_DIR / condition / "Ensemble"
        dst_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {condition}: {len(file_names)} images ===")

        for i, file_name in enumerate(file_names):
            img_path = VAL_IMG_DIR / file_name
            if not img_path.exists():
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            # ── Step 1: YOLOv11 inference ───────────────────────────────────
            r11 = model_v11.predict(str(img_path), device=GPU,
                                    verbose=False, conf=PRE_CONF, iou=0.35)[0]
            boxes11, scores11, labels11 = [], [], []
            for box in (r11.boxes or []):
                xy = box.xyxy[0].cpu().numpy().tolist()
                boxes11.append(xy)
                scores11.append(float(box.conf[0]))
                labels11.append(int(box.cls[0]))

            # ── Early-exit: skip YOLOv26 if v11 is uniformly confident ──────
            early_exit = (
                len(scores11) >= EARLY_EXIT_MIN_DETS
                and all(s >= EARLY_EXIT_CONF for s in scores11)
            )

            if early_exit:
                all_preds = [(boxes11, scores11, labels11)]
            else:
                r26 = model_v26.predict(str(img_path), device=GPU,
                                        verbose=False, conf=PRE_CONF, iou=0.35)[0]
                boxes26, scores26, labels26 = [], [], []
                for box in (r26.boxes or []):
                    xy = box.xyxy[0].cpu().numpy().tolist()
                    boxes26.append(xy)
                    scores26.append(float(box.conf[0]))
                    labels26.append(int(box.cls[0]))
                all_preds = [
                    (boxes11, scores11, labels11),
                    (boxes26, scores26, labels26),
                ]

            # ── Step 2: Per-class WBF fusion (or pass-through if early-exit) ─
            fused = wbf_fuse(all_preds, img_w, img_h) if not early_exit \
                else [{"box": b, "score": s, "cls": l}
                      for b, s, l in zip(boxes11, scores11, labels11)]

            # ── Step 3: Adaptive 8-rule post-processing ─────────────────────
            final_boxes, stats = apply_all_rules(fused, condition, img_w, img_h)
            stats["raw"] = len(fused)

            # ── Step 4: Save annotated image ────────────────────────────────
            out_img = draw_boxes(image.copy(), final_boxes, condition, stats)
            cv2.imwrite(str(dst_dir / file_name), out_img)

            total_images += 1
            total_raw += len(fused)
            total_final += len(final_boxes)

            csv_rows.append({
                "condition": condition,
                "file": file_name,
                "wbf_raw": len(fused),
                "after_conf": stats.get("after_conf", ""),
                "after_area": stats.get("after_area", ""),
                "after_aspect": stats.get("after_aspect", ""),
                "after_large_suppress": stats.get("after_large_suppress", ""),
                "after_cross_nms": stats.get("after_cross_nms", ""),
                "after_vertical": stats.get("after_vertical", ""),
                "after_ppe_anchor": stats.get("after_ppe_anchor", ""),
                "final": stats.get("final", len(final_boxes)),
            })

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(file_names)}] done")

        print(f"  {condition} complete -> {dst_dir}")

    # ── CSV summary ─────────────────────────────────────────────────────────────
    csv_path = LOG_DIR / "ensemble_batch_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    reduction_pct = (1 - total_final / max(total_raw, 1)) * 100
    print(f"\n{'='*60}")
    print(f"ENSEMBLE BATCH COMPLETE")
    print(f"  Images processed : {total_images}")
    print(f"  WBF raw total    : {total_raw}")
    print(f"  Final total      : {total_final}")
    print(f"  Reduction        : {reduction_pct:.1f}%")
    print(f"  Output           : {OUT_DIR}")
    print(f"  CSV              : {csv_path}")
    print(f"{'='*60}")

    # ── ntfy notification ────────────────────────────────────────────────────────
    import subprocess
    subprocess.run([
        "curl", "-s", "-X", "POST",
        "-H", "Title: TONI: Ensemble Batch Complete",
        "-H", "Priority: default",
        "-d", (
            f"Phase 3 WBF ensemble done. "
            f"{total_images} images, {total_raw}->{total_final} detections "
            f"({reduction_pct:.1f}% reduction). "
            f"Output: val_annotated_ensemble/"
        ),
        "ntfy.sh/buildsight-tournament-2026"
    ], check=False)


if __name__ == "__main__":
    run()
