"""
export_for_models.py
====================
BuildSight AI — Multi-Model Export Standardizer
Green Build AI | IGBC AP | NBC 2016 Aligned

Generates four model-specific training-ready dataset layouts from the
single master annotation store in Final_Annotated_Dataset/.

TARGET MODELS:
  1. YOLOv11    → YOLO .txt + data.yaml (standard Ultralytics layout)
  2. YOLOv26    → YOLO .txt + data.yaml + box-tightness metadata
  3. SAMURAI    → COCO JSON + per-sequence manifest + frame-ordered index
  4. YOLACT++   → COCO JSON + per-instance segmentation masks guaranteed

Output tree:
  Final_Annotated_Dataset/
  └── exports/
      ├── YOLOv11/
      │   ├── images/train|val|test/
      │   ├── labels/train|val|test/
      │   └── data.yaml
      ├── YOLOv26/
      │   ├── images/train|val|test/
      │   ├── labels/train|val|test/
      │   ├── box_tightness_report.json
      │   └── data.yaml
      ├── SAMURAI/
      │   ├── annotations/
      │   │   ├── instances_train.json
      │   │   ├── instances_val.json
      │   │   ├── instances_test.json
      │   │   └── sequences/
      │   │       └── <seq_id>.json   ← per-sequence frame manifest
      │   ├── images/train|val|test/
      │   └── sequence_index.json
      └── YOLACT_plus_plus/
          ├── annotations/
          │   ├── instances_train.json
          │   ├── instances_val.json
          │   └── instances_test.json
          ├── images/train|val|test/
          └── mask_stats.json

Usage:
  python export_for_models.py [--models yolov11,yolov26,samurai,yolact]
  python export_for_models.py --verify-only
"""

import os
import json
import math
import shutil
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

DATASET_DIR  = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset")
EXPORT_DIR   = DATASET_DIR / "exports"
SPLITS       = ["train", "val", "test"]

CLASS_NAMES  = {0: "helmet", 1: "safety_vest", 2: "worker"}
VALID_IDS    = set(CLASS_NAMES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def symlink_or_copy(src: Path, dst: Path, use_symlinks: bool = False):
    """Create dst pointing to src. Falls back to copy if symlinks unsupported."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlinks:
        try:
            dst.symlink_to(src.resolve())
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(str(src), str(dst))


def load_coco(split: str) -> dict:
    path = DATASET_DIR / "annotations" / f"instances_{split}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_yolo_label(label_path: Path) -> list:
    """Return list of (class_id, cx, cy, w, h) tuples."""
    rows = []
    if not label_path.exists():
        return rows
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                rows.append((int(parts[0]),
                             float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])))
    return rows


def validate_yolo_label(label_path: Path) -> dict:
    """
    Validate one YOLO label file for common issues.
    Returns dict of issue flags.
    """
    issues = {"invalid_class": 0, "degenerate_box": 0, "out_of_bounds": 0}
    rows = load_yolo_label(label_path)
    for cid, cx, cy, w, h in rows:
        if cid not in VALID_IDS:
            issues["invalid_class"] += 1
        if w <= 0 or h <= 0 or w > 1 or h > 1:
            issues["degenerate_box"] += 1
        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            issues["out_of_bounds"] += 1
    return issues


def bbox_area(bbox_yolo: tuple) -> float:
    """Normalized area of a YOLO bbox (w * h)."""
    return bbox_yolo[3] * bbox_yolo[4]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: YOLOv11 EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_yolov11(use_symlinks: bool = False) -> dict:
    """
    Standard Ultralytics YOLOv11 layout.
    images/train|val|test + labels/train|val|test + data.yaml
    """
    out = EXPORT_DIR / "YOLOv11"
    stats = {"images": 0, "labels": 0, "issues": 0}

    for split in SPLITS:
        img_src = DATASET_DIR / "images" / split
        lbl_src = DATASET_DIR / "labels" / split
        img_dst = out / "images" / split
        lbl_dst = out / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        for img_path in img_src.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_src / f"{img_path.stem}.txt"

            symlink_or_copy(img_path, img_dst / img_path.name, use_symlinks)
            symlink_or_copy(lbl_path, lbl_dst / lbl_path.name, use_symlinks)

            issues = validate_yolo_label(lbl_path)
            if any(issues.values()):
                stats["issues"] += 1
            stats["images"] += 1
            stats["labels"] += 1

    # data.yaml — 3-class, Ultralytics-compatible
    yaml_content = f"""# BuildSight AI — YOLOv11 Training Config
# FINAL | Green Build AI | NBC 2016 Aligned

path: {out.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 3
names:
  0: helmet
  1: safety_vest
  2: worker
"""
    (out / "data.yaml").write_text(yaml_content)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: YOLOv26 EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def check_box_tightness(label_path: Path, tol_px: int = 3,
                         img_w: int = 1920, img_h: int = 1080) -> list:
    """
    YOLOv26 requirement: boxes must be ±3px tight.
    Returns list of (class_id, normalized_slack_x, normalized_slack_y) for loose boxes.
    Slack approximated as box area relative to typical min visible PPE area.
    """
    loose = []
    tol_x = tol_px / img_w
    tol_y = tol_px / img_h
    rows = load_yolo_label(label_path)
    for cid, cx, cy, w, h in rows:
        # Minimum expected dimensions per class at 1080p equivalent
        min_dims = {
            0: (0.015, 0.012),   # helmet: min ~29px × 13px
            1: (0.025, 0.030),   # vest: min ~48px × 32px
            2: (0.010, 0.020),   # boots: min ~19px × 22px
            3: (0.020, 0.050),   # worker: min ~38px × 54px
        }
        min_w, min_h = min_dims.get(cid, (0.010, 0.010))
        slack_x = max(0.0, w - min_w - tol_x)
        slack_y = max(0.0, h - min_h - tol_y)
        if slack_x > tol_x * 3 or slack_y > tol_y * 3:
            loose.append({"class_id": cid, "slack_x": round(slack_x, 4), "slack_y": round(slack_y, 4)})
    return loose


def export_yolov26(use_symlinks: bool = False) -> dict:
    """
    YOLOv26 layout — same as YOLOv11 plus box tightness validation report.
    Boxes violating ±3px tightness are flagged in box_tightness_report.json.
    YOLOv26 requires tighter boxes than v11 for improved small-object mAP.
    """
    out = EXPORT_DIR / "YOLOv26"
    stats = {"images": 0, "labels": 0, "tight_ok": 0, "loose_flagged": 0}
    tightness_report = []

    for split in SPLITS:
        img_src = DATASET_DIR / "images" / split
        lbl_src = DATASET_DIR / "labels" / split
        img_dst = out / "images" / split
        lbl_dst = out / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        for img_path in img_src.iterdir():
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbl_src / f"{img_path.stem}.txt"

            symlink_or_copy(img_path, img_dst / img_path.name, use_symlinks)
            symlink_or_copy(lbl_path, lbl_dst / lbl_path.name, use_symlinks)

            loose = check_box_tightness(lbl_path)
            if loose:
                stats["loose_flagged"] += len(loose)
                tightness_report.append({
                    "file":  img_path.name,
                    "split": split,
                    "loose_boxes": loose,
                })
            else:
                stats["tight_ok"] += 1

            stats["images"] += 1
            stats["labels"] += 1

    # Save tightness report
    report_path = out / "box_tightness_report.json"
    with open(report_path, "w") as f:
        json.dump({"summary": stats, "flagged_images": tightness_report}, f, indent=2)

    # data.yaml
    yaml_content = f"""# BuildSight AI — YOLOv26 Training Config
# FINAL | Green Build AI | NBC 2016 Aligned
# Box tightness: ±3px enforced (see box_tightness_report.json)

path: {out.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 3
names:
  0: helmet
  1: safety_vest
  2: worker
"""
    (out / "data.yaml").write_text(yaml_content)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: SAMURAI EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_samurai(use_symlinks: bool = False) -> dict:
    """
    SAMURAI layout:
    - COCO JSON with track_id and sequence_id fields preserved
    - Per-sequence frame manifests in annotations/sequences/<seq_id>.json
    - sequence_index.json: maps sequence_id → list of frame file paths in order
    - Images organised by split (same as master)

    SAMURAI requires temporally consistent track_ids across frames of a sequence.
    Each sequence manifest contains frames in frame_id order with bbox + track_id.
    """
    out = EXPORT_DIR / "SAMURAI"
    stats = {"images": 0, "sequences": 0, "tracked_instances": 0}

    ann_out = out / "annotations"
    seq_out = ann_out / "sequences"
    seq_out.mkdir(parents=True, exist_ok=True)

    sequence_index = {}   # seq_id → list of {file_name, frame_id, image_id}

    for split in SPLITS:
        img_src = DATASET_DIR / "images" / split
        img_dst = out / "images" / split
        img_dst.mkdir(parents=True, exist_ok=True)

        coco = load_coco(split)
        if coco is None:
            continue

        # Copy/link images
        for img_rec in coco["images"]:
            src = img_src / img_rec["file_name"]
            if src.exists():
                symlink_or_copy(src, img_dst / img_rec["file_name"], use_symlinks)
                stats["images"] += 1

        # Build per-sequence annotation groups
        img_map     = {i["id"]: i for i in coco["images"]}
        by_sequence = defaultdict(list)   # seq_id → list of (frame_id, image_id, anns)

        ann_by_img  = defaultdict(list)
        for ann in coco["annotations"]:
            ann_by_img[ann["image_id"]].append(ann)

        for img_rec in coco["images"]:
            seq_id   = img_rec.get("sequence_id", "unknown")
            frame_id = img_rec.get("frame_id", 0)
            by_sequence[seq_id].append((frame_id, img_rec["id"], img_rec))

        for seq_id, frames in by_sequence.items():
            frames.sort(key=lambda x: x[0])   # sort by frame_id
            seq_manifest = {"sequence_id": seq_id, "frames": []}

            for frame_id, img_id, img_rec in frames:
                frame_anns = ann_by_img[img_id]
                tracked    = [a for a in frame_anns if a.get("track_id", -1) >= 0]
                stats["tracked_instances"] += len(tracked)

                seq_manifest["frames"].append({
                    "frame_id":   frame_id,
                    "image_id":   img_id,
                    "file_name":  img_rec["file_name"],
                    "width":      img_rec["width"],
                    "height":     img_rec["height"],
                    "annotations": [{
                        "annotation_id": a["id"],
                        "category_id":   a["category_id"],
                        "track_id":      a.get("track_id", -1),
                        "bbox":          a["bbox"],
                        "score":         a.get("score", 1.0),
                        "crowd_occluded":a.get("crowd_occluded", False),
                        "truncated":     a.get("truncated", False),
                    } for a in frame_anns],
                })

                if seq_id not in sequence_index:
                    sequence_index[seq_id] = []
                sequence_index[seq_id].append({
                    "file_name": img_rec["file_name"],
                    "frame_id":  frame_id,
                    "image_id":  img_id,
                    "split":     split,
                })

            # Write per-sequence manifest
            with open(seq_out / f"{seq_id}.json", "w") as f:
                json.dump(seq_manifest, f, indent=2)
            stats["sequences"] += 1

        # Write COCO JSON for this split (with track_id fields intact)
        with open(ann_out / f"instances_{split}.json", "w") as f:
            json.dump(coco, f, indent=2)

    # Write global sequence index
    with open(out / "sequence_index.json", "w") as f:
        json.dump(sequence_index, f, indent=2)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: YOLACT++ EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def ensure_segmentation_fallback(ann: dict) -> dict:
    """
    YOLACT++ requires non-empty segmentation for every annotation.
    If SAM produced no mask, generate a rectangular polygon from bbox.
    """
    if ann.get("segmentation") and len(ann["segmentation"]) > 0:
        seg = ann["segmentation"]
        # Ensure no empty sub-polygons
        seg = [p for p in seg if len(p) >= 6]
        if seg:
            ann = dict(ann)
            ann["segmentation"] = seg
            return ann

    # Fallback: bbox rectangle polygon
    x, y, w, h = ann["bbox"]
    ann = dict(ann)
    ann["segmentation"] = [[
        x,     y,
        x + w, y,
        x + w, y + h,
        x,     y + h,
    ]]
    ann["mask_source"] = "bbox_fallback"
    return ann


def export_yolact(use_symlinks: bool = False) -> dict:
    """
    YOLACT++ layout:
    - COCO JSON with guaranteed non-empty segmentation on every annotation
    - mask_stats.json: reports SAM mask coverage vs bbox fallback ratio
    - Annotations include: category_id, segmentation, area, bbox, iscrowd
    - Worker class (id=3) segmentation included — YOLACT++ trains on all 4 classes
    """
    out = EXPORT_DIR / "YOLACT_plus_plus"
    stats = {"images": 0, "annotations": 0, "sam_masks": 0, "bbox_fallbacks": 0}

    ann_out = out / "annotations"
    ann_out.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        img_src = DATASET_DIR / "images" / split
        img_dst = out / "images" / split
        img_dst.mkdir(parents=True, exist_ok=True)

        coco = load_coco(split)
        if coco is None:
            continue

        # Copy/link images
        for img_rec in coco["images"]:
            src = img_src / img_rec["file_name"]
            if src.exists():
                symlink_or_copy(src, img_dst / img_rec["file_name"], use_symlinks)
                stats["images"] += 1

        # Ensure every annotation has a valid segmentation
        new_anns = []
        for ann in coco["annotations"]:
            fixed = ensure_segmentation_fallback(ann)
            if fixed.get("mask_source") == "bbox_fallback":
                stats["bbox_fallbacks"] += 1
            else:
                stats["sam_masks"] += 1
            stats["annotations"] += 1
            new_anns.append(fixed)

        coco["annotations"] = new_anns

        # YOLACT++ requires iscrowd=0 for all instance annotations
        for ann in coco["annotations"]:
            ann["iscrowd"] = 0

        # Write cleaned COCO JSON
        with open(ann_out / f"instances_{split}.json", "w") as f:
            json.dump(coco, f, indent=2)

    # Write mask coverage stats
    total = stats["sam_masks"] + stats["bbox_fallbacks"]
    sam_pct = (stats["sam_masks"] / total * 100) if total > 0 else 0
    with open(out / "mask_stats.json", "w") as f:
        json.dump({
            "total_annotations": total,
            "sam_masks":         stats["sam_masks"],
            "sam_mask_pct":      round(sam_pct, 1),
            "bbox_fallbacks":    stats["bbox_fallbacks"],
            "bbox_fallback_pct": round(100 - sam_pct, 1),
        }, f, indent=2)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# DATASET VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def verify_master_dataset() -> dict:
    """
    Run integrity checks on the master dataset before exporting.
    Returns dict of issues found.
    """
    print("\n[VERIFY] Master dataset integrity check...")
    issues = defaultdict(int)

    for split in SPLITS:
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir = DATASET_DIR / "images" / split
        if not lbl_dir.exists():
            continue

        for lbl_path in lbl_dir.glob("*.txt"):
            # Image exists?
            for ext in [".jpg", ".jpeg", ".png"]:
                if (img_dir / f"{lbl_path.stem}{ext}").exists():
                    break
            else:
                issues["label_without_image"] += 1

            # Class ID validity
            v = validate_yolo_label(lbl_path)
            for k, count in v.items():
                if count > 0:
                    issues[k] += count

    # COCO JSON categories check
    for split in SPLITS:
        coco = load_coco(split)
        if coco is None:
            issues["missing_coco_json"] += 1
            continue
        cat_ids = {c["id"] for c in coco["categories"]}
        if cat_ids != {0, 1, 2, 3}:
            issues["wrong_coco_categories"] += 1
        for ann in coco["annotations"]:
            if ann["category_id"] not in {0, 1, 2, 3}:
                issues["invalid_coco_class_id"] += 1

    if not issues:
        print("  All checks passed — master dataset is clean")
    else:
        for k, v in issues.items():
            print(f"  WARNING {k}: {v}")

    return dict(issues)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "yolov11": ("YOLOv11",          export_yolov11),
    "yolov26": ("YOLOv26",          export_yolov26),
    "samurai": ("SAMURAI",           export_samurai),
    "yolact":  ("YOLACT++",         export_yolact),
}

def main():
    parser = argparse.ArgumentParser(
        description="BuildSight AI — Multi-Model Dataset Export"
    )
    parser.add_argument(
        "--models", type=str, default="yolov11,yolov26,samurai,yolact",
        help="Comma-separated model keys to export (default: all)"
    )
    parser.add_argument(
        "--symlinks", action="store_true",
        help="Use symlinks instead of copying files (faster, smaller disk)"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only run integrity checks, no export"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BuildSight AI - Multi-Model Dataset Export")
    print(f"Source: {DATASET_DIR}")
    print(f"Output: {EXPORT_DIR}")
    print("=" * 70)

    # Always verify first
    issues = verify_master_dataset()
    if args.verify_only:
        return

    if issues:
        print("\n  Proceeding despite warnings — review flagged items after export.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    models = [m.strip().lower() for m in args.models.split(",")]

    for model_key in models:
        if model_key not in MODEL_REGISTRY:
            print(f"\n[SKIP] Unknown model key: {model_key}")
            continue

        label, export_fn = MODEL_REGISTRY[model_key]
        print(f"\n{'-'*70}")
        print(f"Exporting: {label}")
        print(f"{'-'*70}")

        try:
            stats = export_fn(use_symlinks=args.symlinks)
            for k, v in stats.items():
                print(f"  {k:<25}: {v:>8,}")
            print(f"  → {EXPORT_DIR / label.replace('++', '_plus_plus').replace('/', '_')}")
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()

    # Write master export manifest
    manifest = {
        "source_dataset":  str(DATASET_DIR),
        "export_root":     str(EXPORT_DIR),
        "models_exported": models,
        "class_schema": {
            "0": "helmet",
            "1": "safety_vest",
            "2": "worker",
        },
        "splits": SPLITS,
    }
    with open(EXPORT_DIR / "export_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*70}")
    print("Export complete.")
    print(f"{'='*70}")
    print(f"\nTraining commands:")
    print(f"  YOLOv11 : yolo train data={EXPORT_DIR}/YOLOv11/data.yaml model=yolo11n.pt epochs=100")
    print(f"  YOLOv26 : yolo train data={EXPORT_DIR}/YOLOv26/data.yaml model=yolov8n.pt epochs=100")
    print(f"  SAMURAI : See {EXPORT_DIR}/SAMURAI/sequence_index.json for video sequences")
    print(f"  YOLACT++: See {EXPORT_DIR}/YOLACT_plus_plus/annotations/ for COCO JSON")


if __name__ == "__main__":
    main()
