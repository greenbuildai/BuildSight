#!/usr/bin/env python3
"""
find_hard_negatives.py
======================
Scans the Indian Dataset for images likely containing heavy machinery
(excavators, cranes, bulldozers) WITHOUT visible human workers.

These images are the root cause of the safety_vest + worker false positives:
the model saw machinery-heavy scenes during training where annotated workers
were present nearby, and learned to associate machine colors/shapes with PPE.

Adding these images as hard negatives (images the model should output ZERO
detections on) significantly reduces machinery FPs without retraining from scratch.

Output:
  scripts/hard_negatives_report.txt    — list of suspect images to review
  scripts/hard_negatives_preview.jpg   — thumbnail grid for quick visual check

Run locally:
  py -3 scripts/find_hard_negatives.py

Then manually verify the report and move confirmed negatives to:
  Dataset/Hard_Negatives/images/   (empty .txt label files)
and add them to data.yaml train split.
"""

import cv2
import numpy as np
from pathlib import Path

DATASET_ROOTS = [
    Path("e:/Company/Green Build AI/Prototypes/BuildSight/Dataset/Indian Dataset"),
]
VAL_LABEL_DIR  = Path("e:/Company/Green Build AI/Prototypes/BuildSight/Dataset/Final_Annotated_Dataset/labels/val")
TRAIN_LABEL_DIR = Path("e:/Company/Green Build AI/Prototypes/BuildSight/Dataset/Final_Annotated_Dataset/labels/train")
OUTPUT_TXT  = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/hard_negatives_report.txt")
OUTPUT_IMG  = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/hard_negatives_preview.jpg")

# ─── Machinery detection heuristics ──────────────────────────────────────────
# We detect images that likely have heavy machinery by checking:
# 1. Dominant color is yellow/orange (excavators are CAT yellow: HSV ~25-35°, high S/V)
# 2. Large uniform-color blobs (machinery body) in the upper half of the image
# 3. Image is NOT interior (sky / outdoor scene with machinery)

def yellow_fraction(img_bgr):
    """Fraction of pixels in CAT-yellow / high-vis-orange range."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # CAT yellow: H 18-35, S > 100, V > 100
    mask = cv2.inRange(hsv, (18, 100, 100), (35, 255, 255))
    # Also catch orange: H 8-18
    mask2 = cv2.inRange(hsv, (8, 120, 100), (18, 255, 255))
    combined = cv2.bitwise_or(mask, mask2)
    return combined.sum() / (255 * img_bgr.shape[0] * img_bgr.shape[1])


def has_large_yellow_blob(img_bgr, min_blob_area_frac=0.03):
    """True if there's at least one large yellow/orange connected blob."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 100, 100), (38, 255, 255))
    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_px = img_bgr.shape[0] * img_bgr.shape[1]
    for c in contours:
        if cv2.contourArea(c) > min_blob_area_frac * total_px:
            return True, cv2.contourArea(c) / total_px
    return False, 0.0


def is_outdoor_scene(img_bgr):
    """True if the upper 20% of image contains significant sky-blue or bright area."""
    upper = img_bgr[:img_bgr.shape[0]//5, :]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    sky_mask = cv2.inRange(hsv, (100, 30, 100), (130, 255, 255))
    sky_frac = sky_mask.sum() / (255 * upper.shape[0] * upper.shape[1])
    # Also just bright upper region (overexposed sky / dry earth)
    bright_mask = cv2.inRange(upper, (180, 180, 180), (255, 255, 255))
    bright_frac = bright_mask.sum() / (255 * upper.shape[0] * upper.shape[1])
    return (sky_frac + bright_frac) > 0.10


def score_image(img_bgr):
    """Return machinery likelihood score 0-1."""
    yf = yellow_fraction(img_bgr)
    has_blob, blob_frac = has_large_yellow_blob(img_bgr)
    outdoor = is_outdoor_scene(img_bgr)
    score = yf * 3.0 + (blob_frac * 2.0 if has_blob else 0) + (0.2 if outdoor else 0)
    return min(score, 1.0), yf, blob_frac, outdoor


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    suspect = []

    for root in DATASET_ROOTS:
        if not root.exists():
            print(f"[SKIP] {root} not found")
            continue
        img_files = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
        print(f"Scanning {len(img_files)} images in {root.name}...")

        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None or img.shape[0] < 50:
                continue
            img_small = cv2.resize(img, (320, 240))
            score, yf, blob_frac, outdoor = score_image(img_small)

            if score > 0.25:  # suspect threshold
                suspect.append({
                    "path":      str(img_path),
                    "score":     round(score, 3),
                    "yellow_f":  round(yf, 3),
                    "blob_frac": round(blob_frac, 3),
                    "outdoor":   outdoor,
                })

    suspect.sort(key=lambda x: -x["score"])
    print(f"\nFound {len(suspect)} suspect images (score > 0.25)")

    # Write report
    lines = [f"Hard Negative Candidates — BuildSight\n{'='*60}\n",
             f"Total candidates: {len(suspect)}\n\n",
             "Columns: score | yellow_frac | blob_frac | outdoor | path\n\n"]
    for s in suspect:
        lines.append(
            f"{s['score']:.3f}  {s['yellow_f']:.3f}  {s['blob_frac']:.3f}  "
            f"{'Y' if s['outdoor'] else 'N'}  {s['path']}\n"
        )
    OUTPUT_TXT.write_text("".join(lines), encoding="utf-8")
    print(f"Report -> {OUTPUT_TXT}")

    # Build preview grid (top 30)
    top = suspect[:30]
    thumbs = []
    for s in top:
        img = cv2.imread(s["path"])
        if img is None:
            continue
        th = cv2.resize(img, (213, 160))
        label = f"score={s['score']:.2f}  y={s['yellow_f']:.2f}"
        cv2.rectangle(th, (0, 0), (213, 18), (0, 0, 0), -1)
        cv2.putText(th, label, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)
        thumbs.append(th)

    if thumbs:
        while len(thumbs) % 5 != 0:
            thumbs.append(np.zeros((160, 213, 3), np.uint8))
        rows = [np.hstack(thumbs[i:i+5]) for i in range(0, len(thumbs), 5)]
        grid = np.vstack(rows)
        cv2.imwrite(str(OUTPUT_IMG), grid)
        print(f"Preview -> {OUTPUT_IMG}")

    # Summary
    print(f"\nNext step:")
    print(f"  1. Open {OUTPUT_TXT} and visually verify top candidates")
    print(f"  2. For confirmed machinery-only images:")
    print(f"       - Copy image to Dataset/Hard_Negatives/images/")
    print(f"       - Create empty .txt label file (zero detections)")
    print(f"       - Add path to data.yaml train split")
    print(f"  3. Re-train YOLOv11 + YOLOv26 with: yolo train ... epochs=30 freeze=10")
    print(f"     (fine-tune only, don't train from scratch)")


if __name__ == "__main__":
    main()
