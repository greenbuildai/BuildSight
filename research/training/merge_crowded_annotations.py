"""
merge_crowded_annotations.py
============================
Merges newly-annotated Crowded_Condition data (from a --conditions Crowded_Condition run)
into the existing full dataset COCO JSONs that contain Normal/Dusty/LowLight data.

Usage:
  python merge_crowded_annotations.py

Run this AFTER annotate_indian_dataset.py --conditions Crowded_Condition completes.
The script:
  1. Reads existing instances_{split}.json (contains Normal/Dusty/LowLight)
  2. Reads new_crowded_{split}.json  (contains only Crowded_Condition)
  3. Strips old Crowded images/annotations from the existing JSON
  4. Appends new Crowded images/annotations (with re-assigned IDs)
  5. Writes merged result back to instances_{split}.json
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

BASE_DIR   = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight")
OUTPUT_DIR = BASE_DIR / "Dataset" / "Final_Annotated_Dataset"
ANN_DIR    = OUTPUT_DIR / "annotations"
BACKUP_DIR = ANN_DIR / "backup_pre_merge"

CROWDED_CONDITION_KEY = "crowded"   # value of scene_condition in image records


def merge_split(split: str):
    existing_path = ANN_DIR / f"instances_{split}.json"
    crowded_path  = ANN_DIR / f"new_crowded_{split}.json"

    if not existing_path.exists():
        print(f"  [{split}] existing JSON not found — skipping")
        return
    if not crowded_path.exists():
        print(f"  [{split}] new crowded JSON not found — skipping")
        return

    with open(existing_path) as f:
        existing = json.load(f)
    with open(crowded_path) as f:
        crowded = json.load(f)

    # Backup existing before overwriting
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    backup_path = BACKUP_DIR / f"instances_{split}_{ts}.json"
    shutil.copy2(str(existing_path), str(backup_path))
    print(f"  [{split}] Backup → {backup_path.name}")

    # Strip old Crowded images and their annotations from existing
    old_crowded_img_ids = {
        img["id"] for img in existing["images"]
        if img.get("scene_condition") == CROWDED_CONDITION_KEY
    }
    kept_images = [
        img for img in existing["images"]
        if img.get("scene_condition") != CROWDED_CONDITION_KEY
    ]
    kept_annotations = [
        ann for ann in existing["annotations"]
        if ann["image_id"] not in old_crowded_img_ids
    ]

    print(f"  [{split}] Removed {len(old_crowded_img_ids)} old crowded images, "
          f"{len(existing['annotations']) - len(kept_annotations)} old annotations")

    # Determine new ID offsets so new Crowded IDs don't collide
    max_img_id = max((img["id"] for img in kept_images), default=0)
    max_ann_id = max((ann["id"] for ann in kept_annotations), default=0)

    # Re-assign IDs in new crowded data
    img_id_map = {}
    new_images = []
    for img in crowded["images"]:
        new_id = max_img_id + img["id"]
        img_id_map[img["id"]] = new_id
        img_copy = dict(img)
        img_copy["id"] = new_id
        new_images.append(img_copy)

    new_annotations = []
    for ann in crowded["annotations"]:
        new_ann = dict(ann)
        new_ann["id"]       = max_ann_id + ann["id"]
        new_ann["image_id"] = img_id_map.get(ann["image_id"], ann["image_id"])
        new_annotations.append(new_ann)

    print(f"  [{split}] Adding {len(new_images)} new crowded images, "
          f"{len(new_annotations)} new annotations")

    # Build merged JSON
    merged = dict(existing)
    merged["images"]      = kept_images + new_images
    merged["annotations"] = kept_annotations + new_annotations
    merged["info"]["date_created"] = datetime.utcnow().strftime("%Y/%m/%d")

    with open(existing_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"  [{split}] Merged → {len(merged['images'])} images, "
          f"{len(merged['annotations'])} annotations")


def main():
    print("=" * 60)
    print("Merging Crowded_Condition annotations into full dataset")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        merge_split(split)

    print("\nDone. Rename new_crowded_*.json files if you want to keep them.")
    print("Backups saved to:", BACKUP_DIR)


if __name__ == "__main__":
    main()
