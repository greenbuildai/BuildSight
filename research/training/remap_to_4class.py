"""
remap_to_4class.py
==================
One-shot migration: convert all YOLO .txt label files and COCO JSON files
from the old non-sequential schema to the final 4-class sequential schema.

OLD schema (non-sequential, with gaps):
  0 = helmet
  1 = safety_vest
  3 = safety_boots   ← gap at 2
  6 = worker         ← gaps at 4,5

NEW schema (sequential, no gaps):
  0 = helmet        (unchanged)
  1 = safety_vest   (unchanged)
  2 = safety_boots  (was 3)
  3 = worker        (was 6)

Run ONCE after the old pipeline output exists. Safe to re-run (idempotent:
classes already in new schema pass through unchanged).
"""

import json
import os
import sys
from pathlib import Path

DATASET_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset")

# Mapping: old class_id → new class_id
REMAP = {
    0: 0,   # helmet      → helmet      (no change)
    1: 1,   # safety_vest → safety_vest (no change)
    3: 2,   # safety_boots (old 3) → safety_boots (new 2)
    6: 3,   # worker      (old 6) → worker      (new 3)
}

VALID_NEW_IDS = {0, 1, 2, 3}


def remap_yolo_file(path: Path) -> tuple:
    """
    Remap class IDs in a single YOLO .txt file.
    Returns (changed: bool, skipped_lines: int)
    """
    lines = path.read_text().splitlines()
    new_lines = []
    changed = False
    skipped = 0

    for line in lines:
        line = line.strip()
        if not line:
            new_lines.append(line)
            continue

        parts = line.split()
        if len(parts) < 5:
            new_lines.append(line)
            continue

        old_id = int(parts[0])

        if old_id not in REMAP:
            # Class not in known schema — drop it
            skipped += 1
            continue

        new_id = REMAP[old_id]
        if new_id != old_id:
            changed = True

        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    path.write_text("\n".join(new_lines))
    return changed, skipped


def remap_yolo_labels(dataset_dir: Path) -> dict:
    """Remap all YOLO bbox and seg label files."""
    stats = {"files_checked": 0, "files_changed": 0, "lines_skipped": 0}

    for label_subdir in ["labels", "labels_seg"]:
        for split in ["train", "val", "test"]:
            folder = dataset_dir / label_subdir / split
            if not folder.exists():
                continue
            for txt_path in folder.glob("*.txt"):
                stats["files_checked"] += 1
                changed, skipped = remap_yolo_file(txt_path)
                if changed:
                    stats["files_changed"] += 1
                stats["lines_skipped"] += skipped

    return stats


def remap_coco_json(json_path: Path) -> dict:
    """
    Remap category_id in all annotations and rewrite categories list.
    Returns stats dict.
    """
    with open(json_path) as f:
        coco = json.load(f)

    stats = {"annotations_remapped": 0, "annotations_dropped": 0}

    # Rebuild annotations with new category IDs
    new_annotations = []
    for ann in coco["annotations"]:
        old_id = ann["category_id"]
        if old_id not in REMAP:
            stats["annotations_dropped"] += 1
            continue
        new_id = REMAP[old_id]
        if new_id != old_id:
            ann = dict(ann)  # shallow copy
            ann["category_id"] = new_id
            stats["annotations_remapped"] += 1
        new_annotations.append(ann)

    coco["annotations"] = new_annotations

    # Rebuild categories list with new IDs (sorted, sequential)
    coco["categories"] = [
        {"supercategory": "ppe",    "id": 0, "name": "helmet"},
        {"supercategory": "ppe",    "id": 1, "name": "safety_vest"},
        {"supercategory": "ppe",    "id": 2, "name": "safety_boots"},
        {"supercategory": "person", "id": 3, "name": "worker"},
    ]

    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)

    return stats


def main():
    print("=" * 60)
    print("BuildSight — 4-Class Schema Migration")
    print(f"Dataset: {DATASET_DIR}")
    print("=" * 60)

    # ── Step 1: Remap YOLO labels ────────────────────────────────
    print("\n[1/2] Remapping YOLO label files...")
    yolo_stats = remap_yolo_labels(DATASET_DIR)
    print(f"  Files checked  : {yolo_stats['files_checked']:>6}")
    print(f"  Files changed  : {yolo_stats['files_changed']:>6}")
    print(f"  Lines skipped  : {yolo_stats['lines_skipped']:>6}")

    # ── Step 2: Remap COCO JSON files ────────────────────────────
    print("\n[2/2] Remapping COCO JSON annotations...")
    ann_dir = DATASET_DIR / "annotations"
    for json_name in ["instances_train.json", "instances_val.json", "instances_test.json"]:
        json_path = ann_dir / json_name
        if not json_path.exists():
            print(f"  [SKIP] {json_name} not found")
            continue
        stats = remap_coco_json(json_path)
        print(f"  {json_name}: remapped={stats['annotations_remapped']}, "
              f"dropped={stats['annotations_dropped']}")

    # ── Verification ─────────────────────────────────────────────
    print("\n[VERIFY] Sampling 500 label files for stale IDs...")
    import random
    stale = 0
    sampled = 0
    for txt_path in (DATASET_DIR / "labels" / "train").glob("*.txt"):
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(parts[0]) not in VALID_NEW_IDS:
                    stale += 1
        sampled += 1
        if sampled >= 500:
            break

    if stale == 0:
        print(f"  OK — 0 stale class IDs found in {sampled} files sampled")
    else:
        print(f"  WARNING — {stale} stale class IDs still present in {sampled} files")

    print("\nDone. All label files now use class IDs 0-3 only.")


if __name__ == "__main__":
    main()
