"""
merge_all_conditions.py
=======================
Merges Normal + Dusty + Low_Light COCO JSONs into instances_{split}.json
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

ANN_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\annotations")

SOURCES = ["normal", "dusty", "ll"]  # prefixes of saved JSONs

def merge_split(split):
    merged_images = []
    merged_annotations = []
    img_id = 1
    ann_id = 1
    categories = None

    for src in SOURCES:
        src_path = ANN_DIR / f"{src}_{split}.json"
        if not src_path.exists():
            print(f"  WARNING: {src_path.name} not found — skipping")
            continue
        with open(src_path) as f:
            data = json.load(f)

        if categories is None:
            categories = data.get("categories", [])

        # Remap IDs
        id_remap = {}
        for img in data.get("images", []):
            old_id = img["id"]
            img = dict(img)
            img["id"] = img_id
            id_remap[old_id] = img_id
            img_id += 1
            merged_images.append(img)

        for ann in data.get("annotations", []):
            ann = dict(ann)
            ann["id"] = ann_id
            ann["image_id"] = id_remap.get(ann["image_id"], ann["image_id"])
            ann_id += 1
            merged_annotations.append(ann)

        print(f"  [{split}] {src}: {len(data.get('images',[]))} imgs, {len(data.get('annotations',[]))} anns")

    result = {
        "info": {"description": "BuildSight Indian Dataset", "date_created": datetime.now().isoformat()},
        "licenses": [],
        "categories": categories or [],
        "images": merged_images,
        "annotations": merged_annotations,
    }

    out_path = ANN_DIR / f"instances_{split}.json"
    # Backup if exists
    if out_path.exists():
        bak = ANN_DIR / f"instances_{split}.bak.json"
        shutil.copy2(out_path, bak)

    with open(out_path, "w") as f:
        json.dump(result, f)

    print(f"  [{split}] MERGED: {len(merged_images)} images, {len(merged_annotations)} annotations -> {out_path.name}")

if __name__ == "__main__":
    print("Merging Normal + Dusty + Low_Light COCO JSONs...")
    for split in ["train", "val", "test"]:
        merge_split(split)
    print("Done.")
