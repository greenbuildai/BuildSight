#!/usr/bin/env python3
"""
smoke_test_fix.py
=================
Smoke test: runs local YOLO model on 10 Normal_Site_Condition images,
produces side-by-side BEFORE vs AFTER comparison showing the 3 inference fixes:
  FIX-A  Aspect-ratio filter  (excavator landscape boxes removed)
  FIX-B  Cross-class NMS      (duplicate boxes per person removed)
  FIX-C  Per-class confidence (raised thresholds)

Output:
  scripts/smoke_test_before_after.jpg   — 10-row grid: left=before, right=after
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ─── Config ───────────────────────────────────────────────────────────────────
WEIGHTS     = Path("e:/Company/Green Build AI/Prototypes/BuildSight/buildsight-v1.1/Core/backend/models/best.pt")
IMG_DIR     = Path("e:/Company/Green Build AI/Prototypes/BuildSight/Dataset/Indian Dataset/Normal_Site_Condition")
OUTPUT_IMG  = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/smoke_test_before_after.jpg")
N_IMAGES    = 10

# Model class map (v1.1 local model)
CLS_NAMES   = {0: "person", 1: "helmet", 2: "safety-vest"}
CLS_COLORS  = {0: (255, 100, 0), 1: (0, 255, 0), 2: (0, 165, 255)}  # BGR

# BEFORE: original flat threshold (what SASTRA currently uses)
CONF_BEFORE = 0.25

# AFTER: FIX-C per-class thresholds
CONF_AFTER  = {0: 0.42, 1: 0.30, 2: 0.38}   # person, helmet, safety-vest

# FIX-A: worker/person max aspect ratio (landscape = likely machinery)
MAX_PERSON_ASPECT = 1.25

# FIX-B: cross-class IoU suppression
CROSS_IOU = 0.45


# ─── Post-processing ──────────────────────────────────────────────────────────

def iou(a, b):
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-6)


def apply_fixes(raw_boxes):
    # FIX-C
    boxes = [b for b in raw_boxes if b["score"] >= CONF_AFTER[b["cls"]]]

    # FIX-A: aspect ratio for person (class 0)
    filtered = []
    for b in boxes:
        if b["cls"] == 0:
            bx = b["box"]
            w = bx[2] - bx[0]
            h = bx[3] - bx[1]
            if h > 0 and (w / h) > MAX_PERSON_ASPECT:
                continue   # landscape = likely machinery cab
        filtered.append(b)
    boxes = filtered

    # FIX-B: cross-class NMS
    boxes = sorted(boxes, key=lambda x: -x["score"])
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j] or boxes[i]["cls"] == boxes[j]["cls"]:
                continue
            if iou(boxes[i]["box"], boxes[j]["box"]) > CROSS_IOU:
                keep[j] = False

    return [b for b, k in zip(boxes, keep) if k]


def draw_boxes(img, boxes, show_count=True):
    img = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        cls   = b["cls"]
        color = CLS_COLORS[cls]
        label = f"{CLS_NAMES[cls]} {b['score']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if show_count:
        counts = {}
        for b in boxes:
            counts[CLS_NAMES[b["cls"]]] = counts.get(CLS_NAMES[b["cls"]], 0) + 1
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(img, summary, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1)
    return img


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading model: {WEIGHTS.name}")
    model = YOLO(str(WEIGHTS))

    img_files = sorted(IMG_DIR.glob("*.jpg"))
    # Pick a varied subset: every Nth image
    step = max(1, len(img_files) // N_IMAGES)
    selected = img_files[::step][:N_IMAGES]
    print(f"Selected {len(selected)} images from {len(img_files)} total")

    rows = []
    total_before = 0
    total_after  = 0

    for img_path in selected:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (640, 480))

        # Run inference with permissive gate, apply fixes ourselves
        result = model.predict(str(img_path), device="cpu", verbose=False,
                               conf=0.15, iou=0.50)[0]

        raw_boxes = []
        for box in (result.boxes or []):
            xy  = box.xyxy[0].cpu().numpy().tolist()
            raw_boxes.append({
                "box":   xy,
                "cls":   int(box.cls[0]),
                "score": float(box.conf[0]),
            })

        # Resize boxes to 640x480
        orig_h, orig_w = cv2.imread(str(img_path)).shape[:2]
        sx = 640 / orig_w
        sy = 480 / orig_h
        for b in raw_boxes:
            b["box"] = [b["box"][0]*sx, b["box"][1]*sy, b["box"][2]*sx, b["box"][3]*sy]

        # BEFORE: original flat conf filter only
        before = [b for b in raw_boxes if b["score"] >= CONF_BEFORE]
        # AFTER: all 3 fixes
        after  = apply_fixes(raw_boxes)

        total_before += len(before)
        total_after  += len(after)

        before_img = draw_boxes(img, before)
        after_img  = draw_boxes(img, after)

        # Labels
        bl = f"BEFORE  {len(before)} dets  conf>=0.25"
        al = f"AFTER   {len(after)} dets  FIX-A+B+C"
        removed = len(before) - len(after)
        color_b = (0, 80, 200)
        color_a = (0, 180, 0)

        cv2.rectangle(before_img, (0, 454), (640, 480), color_b, -1)
        cv2.putText(before_img, bl, (4, 472), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.rectangle(after_img, (0, 454), (640, 480), color_a, -1)
        cv2.putText(after_img, al + f"  (-{removed})", (4, 472),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Divider
        div = np.full((480, 4, 3), 255, dtype=np.uint8)
        row = np.hstack([before_img, div, after_img])
        rows.append(row)

        print(f"  {img_path.name}: before={len(before)} -> after={len(after)} (-{removed})")

    grid = np.vstack(rows)

    # Header bar
    header = np.zeros((36, grid.shape[1], 3), dtype=np.uint8)
    summary = (f"BuildSight Inference Fix Smoke Test | "
               f"Total detections: BEFORE={total_before}  AFTER={total_after}  "
               f"REMOVED={total_before - total_after} ({100*(total_before-total_after)/max(total_before,1):.0f}%)")
    cv2.putText(header, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
    grid = np.vstack([header, grid])

    cv2.imwrite(str(OUTPUT_IMG), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"\nSaved -> {OUTPUT_IMG}")
    print(f"Total: BEFORE={total_before}  AFTER={total_after}  "
          f"removed={total_before-total_after} ({100*(total_before-total_after)/max(total_before,1):.0f}%)")


if __name__ == "__main__":
    main()
