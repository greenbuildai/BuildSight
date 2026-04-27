"""
validate_annotations.py
========================
BuildSight AI — Annotation QA Validation Script
Green Build AI | IGBC AP | NBC 2016 Aligned

Standalone script — zero external dependencies beyond OpenCV + numpy.

Reads YOLO .txt annotation files from the dataset, validates the
4-class schema, outputs per-image flag report, and prints full
dataset summary statistics with PASS/FAIL verdict.

Usage:
  python validate_annotations.py --dataset_path "E:\\Company\\Green Build AI\\Prototypes\\BuildSight\\Dataset\\Indian Dataset"

  Optional flags:
    --labels_dir   Path to YOLO labels directory (default: auto-discover from dataset_path)
    --verbose      Print per-image table to stdout (default: True)
    --report_path  Write per-image report to CSV file
"""

import os
import sys
import re
import json
import argparse
import csv
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# 3-CLASS SCHEMA (must match pipeline_config.py)
# ─────────────────────────────────────────────────────────────────────────────

CLASS_ID = {
    "helmet":      0,
    "safety_vest": 1,
    "worker":      2,
}
CLASS_NAMES = {v: k for k, v in CLASS_ID.items()}
VALID_CLASS_IDS = set(CLASS_ID.values())   # {0, 1, 2}

# ─────────────────────────────────────────────────────────────────────────────
# CONDITION DETECTION FROM SUBFOLDER NAME
# ─────────────────────────────────────────────────────────────────────────────

FOLDER_TO_CONDITION = {
    "normal_site_condition": "normal",
    "dusty_condition":       "dusty",
    "low_light_condition":   "low_light",
    "crowded_condition":     "crowded",
}

def infer_condition_from_path(path: Path) -> str:
    """
    Walk up the path hierarchy looking for a known condition subfolder name.
    Falls back to 'unknown' if none found.
    """
    for part in reversed(path.parts):
        key = part.lower().replace(" ", "_")
        if key in FOLDER_TO_CONDITION:
            return FOLDER_TO_CONDITION[key]
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo_annotations(label_path: Path) -> list:
    """
    Parse a YOLO .txt annotation file.
    Returns list of dicts: {class_id, cx, cy, w, h}
    Skips malformed lines.
    """
    annotations = []
    if not label_path.exists():
        return annotations

    with open(label_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                cx     = float(parts[1])
                cy     = float(parts[2])
                w      = float(parts[3])
                h      = float(parts[4])
                annotations.append({
                    "class_id": cls_id,
                    "cx": cx, "cy": cy, "w": w, "h": h,
                })
            except ValueError:
                continue

    return annotations


def estimate_frame_area_coverage(annotation: dict) -> float:
    """Return the fraction of the full frame covered by this box (w * h)."""
    return annotation["w"] * annotation["h"]


# ─────────────────────────────────────────────────────────────────────────────
# FLAG EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_flags(img_name: str, condition: str, annotations: list,
                   total_annotation_count: int) -> dict:
    """
    Evaluate all 6 flag conditions for one image.

    Returns dict of flag_name → (triggered: bool, detail: str)
    """
    workers = [a for a in annotations if a["class_id"] == CLASS_ID["worker"]]
    helmets = [a for a in annotations if a["class_id"] == CLASS_ID["helmet"]]
    vests   = [a for a in annotations if a["class_id"] == CLASS_ID["safety_vest"]]

    w_count = len(workers)
    h_count = len(helmets)
    v_count = len(vests)

    flags = {}

    # HELMET_MISSING: workers present but zero helmets
    flags["HELMET_MISSING"] = (
        w_count > 0 and h_count == 0,
        f"workers={w_count} helmets={h_count}"
    )

    # VEST_MISSING: workers present but zero vests
    flags["VEST_MISSING"] = (
        w_count > 0 and v_count == 0,
        f"workers={w_count} vests={v_count}"
    )

    # GROUP_BOX_FOUND: any single box covers >40% of frame in crowded scene
    max_coverage = max((estimate_frame_area_coverage(a) for a in annotations), default=0.0)
    flags["GROUP_BOX_FOUND"] = (
        condition == "crowded" and max_coverage > 0.40,
        f"max_coverage={max_coverage:.2f}"
    )

    # ZERO_DETECTION: entire image has zero annotations
    flags["ZERO_DETECTION"] = (
        total_annotation_count == 0,
        "no annotations in frame"
    )

    # PPE_RATIO_LOW: helmet/worker ratio below 50%
    ppe_ratio = (h_count / w_count) if w_count > 0 else 1.0
    flags["PPE_RATIO_LOW"] = (
        w_count > 0 and ppe_ratio < 0.5,
        f"helmet/worker={ppe_ratio:.2f}"
    )

    return flags, workers, helmets, vests


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def discover_images_and_labels(dataset_path: Path, labels_dir: Path = None):
    """
    Walk dataset_path for images.
    For each image, determine the corresponding YOLO label path:
      1. labels_dir / image_stem.txt  (if labels_dir provided)
      2. <same_folder_as_image> / <image_stem>.txt
      3. Auto-discover Final_Annotated_Dataset/labels/ tree

    Returns list of (image_path, label_path, condition)
    """
    results = []

    # If labels_dir not provided, attempt auto-discover from dataset_path
    buildsight_root = None
    if labels_dir is None:
        # Walk up from dataset_path to find BuildSight root
        candidate = dataset_path
        for _ in range(5):
            candidate = candidate.parent
            output_labels = candidate / "Dataset" / "Final_Annotated_Dataset" / "labels"
            if output_labels.exists():
                buildsight_root = candidate
                break

    for img_path in sorted(dataset_path.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        stem = img_path.stem
        condition = infer_condition_from_path(img_path)

        # Priority 1: explicit labels_dir
        if labels_dir is not None:
            lbl = labels_dir / f"{stem}.txt"
            results.append((img_path, lbl, condition))
            continue

        # Priority 2: Final_Annotated_Dataset labels tree (search all splits)
        if buildsight_root is not None:
            found_label = None
            for split in ("train", "val", "test"):
                candidate_lbl = (
                    buildsight_root
                    / "Dataset" / "Final_Annotated_Dataset"
                    / "labels" / split / f"{stem}.txt"
                )
                if candidate_lbl.exists():
                    found_label = candidate_lbl
                    break
            if found_label:
                results.append((img_path, found_label, condition))
                continue

        # Priority 3: label file alongside image
        sibling_lbl = img_path.parent / f"{stem}.txt"
        results.append((img_path, sibling_lbl, condition))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN VALIDATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(dataset_path: Path, labels_dir: Path = None,
                   verbose: bool = True, report_path: Path = None):
    """
    Full dataset validation run.
    Prints per-image report and summary statistics.
    Returns True if PASS, False if FAIL.
    """

    print(f"\nBuildSight AI — Annotation Validation")
    print(f"Dataset path : {dataset_path}")
    print(f"Labels dir   : {labels_dir or '(auto-discover)'}")
    print("=" * 100)

    image_label_pairs = discover_images_and_labels(dataset_path, labels_dir)

    if not image_label_pairs:
        print("[ERROR] No images found under dataset path.")
        return False

    print(f"Images found: {len(image_label_pairs)}\n")

    # ── Per-image results collection ────────────────────────────────────────
    results = []

    # Aggregate counters
    total_images      = 0
    total_annotations = 0
    class_totals      = defaultdict(int)   # class_id → count

    per_condition = defaultdict(lambda: {
        "images": 0, "workers": 0, "helmets": 0, "vests": 0
    })

    flag_counts = defaultdict(int)   # flag_name → image count

    # Column widths for tabular output
    COL_IMG  = 40
    COL_COND = 10
    COL_NUM  = 7

    if verbose:
        header = (
            f"{'image_name':<{COL_IMG}} | {'condition':<{COL_COND}} | "
            f"{'workers':>{COL_NUM}} | {'helmets':>{COL_NUM}} | "
            f"{'vests':>{COL_NUM}} | flags"
        )
        separator = "-" * len(header)
        print(header)
        print(separator)

    report_rows = []

    for img_path, lbl_path, condition in image_label_pairs:
        annotations      = load_yolo_annotations(lbl_path)
        ann_count        = len(annotations)
        total_images     += 1
        total_annotations += ann_count

        # Count per class
        for ann in annotations:
            cid = ann["class_id"]
            if cid in VALID_CLASS_IDS:
                class_totals[cid] += 1

        flags_result, workers, helmets, vests = evaluate_flags(
            img_path.name, condition, annotations, ann_count
        )

        w_count = len(workers)
        h_count = len(helmets)
        v_count = len(vests)

        per_condition[condition]["images"]  += 1
        per_condition[condition]["workers"] += w_count
        per_condition[condition]["helmets"] += h_count
        per_condition[condition]["vests"]   += v_count

        # Build flag string
        triggered_flags = []
        for flag_name, (triggered, detail) in flags_result.items():
            if triggered:
                flag_counts[flag_name] += 1
                # Append count suffix for MISSING flags
                if flag_name in ("HELMET_MISSING", "VEST_MISSING"):
                    deficit = w_count - (
                        h_count if "HELMET" in flag_name else
                        v_count
                    )
                    if deficit > 1:
                        triggered_flags.append(f"{flag_name}({deficit})")
                    else:
                        triggered_flags.append(flag_name)
                else:
                    triggered_flags.append(flag_name)

        flags_str = " ".join(triggered_flags) if triggered_flags else "OK"

        img_name_short = img_path.name[:COL_IMG]

        if verbose:
            row = (
                f"{img_name_short:<{COL_IMG}} | {condition:<{COL_COND}} | "
                f"{w_count:>{COL_NUM}} | {h_count:>{COL_NUM}} | "
                f"{v_count:>{COL_NUM}} | {flags_str}"
            )
            print(row)

        report_rows.append({
            "image_name": img_path.name,
            "condition":  condition,
            "workers":    w_count,
            "helmets":    h_count,
            "vests":      v_count,
            "flags":      flags_str,
        })

    # ── Save CSV report ──────────────────────────────────────────────────────
    if report_path is not None:
        with open(report_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "image_name", "condition", "workers", "helmets",
                "vests", "flags"
            ])
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"\nPer-image report saved → {report_path}")

    # ── Summary statistics ───────────────────────────────────────────────────
    print("\n")
    print("=" * 60)
    print("DATASET SUMMARY — BuildSight AI Indian Dataset")
    print("=" * 60)
    print(f"Total images        : {total_images:>6}")
    print(f"Total annotations   : {total_annotations:>6}")

    for cid in sorted(VALID_CLASS_IDS):
        cname  = CLASS_NAMES[cid]
        count  = class_totals[cid]
        pct    = (count / total_annotations * 100) if total_annotations > 0 else 0.0
        print(f"  → {cname:<16} : {count:>6}  ({pct:5.1f}%)")

    print("\nPer-condition breakdown:")
    for cond in ("normal", "dusty", "low_light", "crowded", "unknown"):
        d = per_condition[cond]
        if d["images"] == 0:
            continue
        print(
            f"  {cond:<10} : {d['images']:>4} images | "
            f"{d['workers']:>4} workers | "
            f"{d['helmets']:>4} helmets | "
            f"{d['vests']:>4} vests"
        )

    print("\nFlag summary:")
    flag_order = [
        "HELMET_MISSING", "VEST_MISSING",
        "GROUP_BOX_FOUND", "ZERO_DETECTION", "PPE_RATIO_LOW"
    ]
    for flag_name in flag_order:
        count = flag_counts[flag_name]
        pct   = (count / total_images * 100) if total_images > 0 else 0.0
        print(f"  {flag_name:<20} : {count:>4} images ({pct:5.1f}%)")

    # ── PASS/FAIL verdict ────────────────────────────────────────────────────
    helmet_missing_pct    = (flag_counts["HELMET_MISSING"] / total_images * 100) if total_images > 0 else 0.0
    vest_missing_pct      = (flag_counts["VEST_MISSING"]   / total_images * 100) if total_images > 0 else 0.0
    zero_detection_pct    = (flag_counts["ZERO_DETECTION"] / total_images * 100) if total_images > 0 else 0.0

    pass_criteria = {
        "HELMET_MISSING < 10%": helmet_missing_pct < 10.0,
        "VEST_MISSING < 10%":   vest_missing_pct   < 10.0,
        "ZERO_DETECTION < 5%":  zero_detection_pct  < 5.0,
    }
    verdict = all(pass_criteria.values())

    print("\n" + "=" * 60)
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}")
    print("  Pass criteria:")
    for criterion, passed in pass_criteria.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {criterion}")
    print("=" * 60)

    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BuildSight AI — Annotation QA Validation Script"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset",
        help="Path to the Indian Dataset root folder containing condition subfolders"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default=None,
        help="Explicit path to YOLO labels directory. If omitted, auto-discovers from dataset_path."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-image report table to stdout (default: True)"
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Write per-image report to this CSV file path"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    labels_dir   = Path(args.labels_dir) if args.labels_dir else None
    report_path  = Path(args.report_path) if args.report_path else None

    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    result = run_validation(
        dataset_path = dataset_path,
        labels_dir   = labels_dir,
        verbose      = args.verbose,
        report_path  = report_path,
    )

    sys.exit(0 if result else 1)
