"""
BuildSight Annotation Visualizer
Generates preview grids showing bounding boxes + segmentation polygons
across all 4 site conditions for visual QA.
"""
import os
import cv2
import json
import random
import numpy as np
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight")
OUTPUT_DIR = BASE_DIR / "Dataset" / "Final_Annotated_Dataset"
JSON_PATH  = OUTPUT_DIR / "annotations" / "instances_train.json"
IMG_DIR    = OUTPUT_DIR / "images" / "train"
PREVIEW_OUT = BASE_DIR / "Dataset" / "annotation_preview.png"

# Colors per class (BGR for cv2) — 3-class sequential schema
CLASS_COLORS = {
    0: (0,   200, 255),   # helmet      — amber/yellow
    1: (0,   220,  80),   # safety_vest — green
    2: (200,  40, 220),   # worker      — magenta
}
DEFAULT_COLOR = (255, 255, 255)

CONDITIONS = [
    "Normal_Site_Condition",
    "Crowded_Condition",
    "Dusty_Condition",
    "Low_Light_Condition",
]


def draw_annotations(img, anns, categories):
    """Draw bboxes + segmentation polygons on an image."""
    overlay = img.copy()

    for ann in anns:
        cat_id = ann["category_id"]
        cat_name = categories.get(cat_id, f"cls_{cat_id}")
        color = CLASS_COLORS.get(cat_id, DEFAULT_COLOR)
        score = ann.get("score", 1.0)

        # Draw segmentation polygon
        if "segmentation" in ann and ann["segmentation"]:
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                if len(poly) >= 3:
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(img, [poly], True, color, 2)

        # Draw bounding box
        bbox = ann["bbox"]  # [x, y, w, h]
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Label — class name only, no confidence score
        label = cat_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Blend transparent polygon fill
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    return img


def make_preview_grid(rows=4, cols=3, cell_w=640, cell_h=480):
    """Generate a grid preview: one row per condition, N random samples each."""
    print(f"Loading COCO JSON: {JSON_PATH}")
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}

    # Group annotations by image_id
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Group images by condition
    imgs_by_cond = {}
    for img in data["images"]:
        cond = img.get("scene_condition", "unknown")
        imgs_by_cond.setdefault(cond, []).append(img)

    # Create canvas
    grid_h = rows * cell_h + (rows + 1) * 10
    grid_w = cols * cell_w + (cols + 1) * 10
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 30  # dark background

    for row_idx, condition in enumerate(CONDITIONS):
        cond_images = imgs_by_cond.get(condition, [])
        if not cond_images:
            print(f"  No images for {condition}, skipping")
            continue

        # Pick images that have annotations
        annotated = [im for im in cond_images if im["id"] in anns_by_img]
        if not annotated:
            annotated = cond_images

        samples = random.sample(annotated, min(cols, len(annotated)))

        for col_idx, img_info in enumerate(samples):
            img_path = IMG_DIR / img_info["file_name"]
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Draw annotations
            anns = anns_by_img.get(img_info["id"], [])
            img = draw_annotations(img, anns, categories)

            # Add condition label on the image
            label = f"{condition} ({len(anns)} objects)"
            cv2.putText(img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Resize to cell
            img_resized = cv2.resize(img, (cell_w, cell_h))

            # Place on canvas
            y0 = row_idx * cell_h + (row_idx + 1) * 10
            x0 = col_idx * cell_w + (col_idx + 1) * 10
            canvas[y0:y0 + cell_h, x0:x0 + cell_w] = img_resized

        print(f"  {condition}: {len(samples)} samples drawn")

    # Save
    cv2.imwrite(str(PREVIEW_OUT), canvas)
    print(f"\nPreview saved to: {PREVIEW_OUT}")
    print(f"   Grid: {rows} conditions × {cols} samples = {rows * cols} cells")
    return str(PREVIEW_OUT)


def save_individual_samples(n_per_condition=3, out_dir=None):
    """Also save individual full-res annotated images for close inspection."""
    if out_dir is None:
        out_dir = BASE_DIR / "Dataset" / "annotation_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading COCO JSON for individual samples...")
    with open(JSON_PATH) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    imgs_by_cond = {}
    for img in data["images"]:
        cond = img.get("scene_condition", "unknown")
        imgs_by_cond.setdefault(cond, []).append(img)

    saved = 0
    for condition in CONDITIONS:
        cond_imgs = imgs_by_cond.get(condition, [])
        annotated = [im for im in cond_imgs if im["id"] in anns_by_img]
        if not annotated:
            continue
        # Pick a mix: highest annotation count + random
        scored = sorted(annotated, key=lambda x: -len(anns_by_img.get(x["id"], [])))
        picks = scored[:n_per_condition]

        for img_info in picks:
            img_path = IMG_DIR / img_info["file_name"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            anns = anns_by_img.get(img_info["id"], [])
            img = draw_annotations(img, anns, categories)
            n_anns = len(anns)
            fname = f"{condition}_{img_info['file_name'].replace('.jpg','')}_n{n_anns}.jpg"
            cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 93])
            print(f"  {fname}")
            saved += 1

    print(f"\n{saved} individual samples saved to:\n  {out_dir}")


if __name__ == "__main__":
    random.seed(42)  # Reproducible samples
    make_preview_grid(rows=4, cols=3, cell_w=640, cell_h=480)
    save_individual_samples(n_per_condition=3)
