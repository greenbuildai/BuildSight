"""
phash_dedup_and_blur.py
=======================
Two-pass pre-annotation QA for the BuildSight Indian Dataset:

Pass 1 — pHash Near-Duplicate Detection
  Finds visually near-identical images using perceptual hashing.
  Hamming distance threshold: 8 (0 = exact pixel match, higher = more similar).
  For burst sequences / video frames, this removes redundant frames while
  keeping enough temporal diversity for SAMURAI training.

  Strategy within a pHash cluster:
    - Keep the sharpest image (highest Laplacian variance)
    - Remove the rest

Pass 2 — Blur Detection
  Computes Laplacian variance per image.
  Images with variance < BLUR_THRESHOLD are flagged.
  By default, these are REPORTED only (not deleted) so you can review them.
  Set REMOVE_BLURRY = True to auto-delete.

Usage:
  python scripts/phash_dedup_and_blur.py
  python scripts/phash_dedup_and_blur.py --dry-run        # report only, no deletes
  python scripts/phash_dedup_and_blur.py --remove-blurry  # also delete blurry images
  python scripts/phash_dedup_and_blur.py --condition Crowded_Condition  # single folder
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

try:
    import imagehash
except ImportError:
    raise ImportError("Run: pip install imagehash")

# -----------------------------------------------------------------------------
DATA_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")

CONDITIONS = [
    "Normal_Site_Condition",
    "Crowded_Condition",
    "Dusty_Condition",
    "Low_Light_Condition",
]

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# pHash Hamming distance threshold.
# 0  = exact duplicate (same as MD5 match)
# 8  = very similar (recommended for burst frames)
# 12 = similar scenes, different angle
PHASH_THRESHOLD = 8

# Laplacian variance below this = blurry.
# 100 is standard for construction site imagery.
BLUR_THRESHOLD = 100.0

# -----------------------------------------------------------------------------

def laplacian_variance(img_path: str) -> float:
    """Compute Laplacian variance (sharpness metric). Higher = sharper."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_phash(img_path: str):
    """Compute perceptual hash. Returns None on error."""
    try:
        with Image.open(img_path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def process_condition(condition: str, dry_run: bool, remove_blurry: bool) -> dict:
    folder = DATA_DIR / condition
    if not folder.exists():
        print(f"  [SKIP] Not found: {folder}")
        return {}

    files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ])

    print(f"\n{'-'*60}")
    print(f"  {condition}: {len(files)} images")
    print(f"{'-'*60}")

    # -- Pass 1: pHash clustering ---------------------------------------------
    print("  [Pass 1] Computing pHashes...")
    hashes = {}       # path → phash
    sharpness = {}    # path → laplacian variance

    for f in tqdm(files, desc="  pHash", leave=False):
        ph = compute_phash(str(f))
        if ph is not None:
            hashes[f] = ph
            sharpness[f] = laplacian_variance(str(f))

    # Greedy clustering: assign each image to first cluster within threshold
    clusters = []           # list of [list of Path]
    assigned = set()

    sorted_paths = list(hashes.keys())
    for i, path_a in enumerate(sorted_paths):
        if path_a in assigned:
            continue
        cluster = [path_a]
        assigned.add(path_a)
        ph_a = hashes[path_a]
        for path_b in sorted_paths[i + 1:]:
            if path_b in assigned:
                continue
            if (ph_a - hashes[path_b]) <= PHASH_THRESHOLD:
                cluster.append(path_b)
                assigned.add(path_b)
        if len(cluster) > 1:
            clusters.append(cluster)

    # Identify duplicates: within each cluster, keep sharpest, remove rest
    to_remove_dup = []
    for cluster in clusters:
        # Sort by sharpness descending — keep index 0
        cluster_sorted = sorted(cluster, key=lambda p: sharpness.get(p, 0.0), reverse=True)
        keeper = cluster_sorted[0]
        removals = cluster_sorted[1:]
        to_remove_dup.extend(removals)

    print(f"  [Pass 1] Near-duplicate clusters found: {len(clusters)}")
    print(f"  [Pass 1] Images to remove: {len(to_remove_dup)}")

    # -- Pass 2: Blur detection ------------------------------------------------
    print("  [Pass 2] Checking blur...")
    blurry = [
        f for f, score in sharpness.items()
        if score < BLUR_THRESHOLD and f not in set(to_remove_dup)
    ]
    print(f"  [Pass 2] Blurry images (Laplacian < {BLUR_THRESHOLD}): {len(blurry)}")

    # -- Execute ---------------------------------------------------------------
    removed_dup   = 0
    removed_blur  = 0

    if not dry_run:
        for f in to_remove_dup:
            try:
                f.unlink()
                removed_dup += 1
            except Exception as e:
                print(f"  [WARN] Could not remove {f.name}: {e}")

        if remove_blurry:
            for f in blurry:
                try:
                    f.unlink()
                    removed_blur += 1
                except Exception as e:
                    print(f"  [WARN] Could not remove {f.name}: {e}")
    else:
        print("  [DRY RUN] No files deleted.")
        if to_remove_dup:
            print(f"  Sample near-duplicates that WOULD be removed:")
            for f in to_remove_dup[:5]:
                print(f"    - {f.name}  (sharpness: {sharpness.get(f, 0):.1f})")
        if blurry and remove_blurry:
            print(f"  Sample blurry images that WOULD be removed:")
            for f in blurry[:5]:
                print(f"    - {f.name}  (Laplacian: {sharpness.get(f, 0):.1f})")

    # Blur report (always, even without removal)
    if blurry and not remove_blurry:
        print(f"  [INFO] Blurry images retained (use --remove-blurry to delete):")
        for f in blurry[:10]:
            print(f"    - {f.name}  (Laplacian: {sharpness.get(f, 0):.1f})")
        if len(blurry) > 10:
            print(f"    ... and {len(blurry) - 10} more")

    remaining = len(files) - removed_dup - removed_blur
    return {
        "condition":      condition,
        "scanned":        len(files),
        "dup_clusters":   len(clusters),
        "removed_dup":    removed_dup if not dry_run else f"{len(to_remove_dup)} (dry-run)",
        "blurry_found":   len(blurry),
        "removed_blur":   removed_blur if not dry_run else 0,
        "remaining":      remaining if not dry_run else len(files),
    }


def main():
    parser = argparse.ArgumentParser(description="pHash dedup + blur detection")
    parser.add_argument("--dry-run",       action="store_true", help="Report only, no deletes")
    parser.add_argument("--remove-blurry", action="store_true", help="Also delete blurry images")
    parser.add_argument("--condition",     type=str, default="", help="Process single condition only")
    args = parser.parse_args()

    conditions = [args.condition] if args.condition else CONDITIONS
    results = []

    print(f"\n{'='*60}")
    print("  BuildSight - pHash Dedup + Blur Detection")
    print(f"  Threshold: Hamming <= {PHASH_THRESHOLD} | Blur < {BLUR_THRESHOLD}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"{'='*60}")

    for condition in conditions:
        result = process_condition(condition, args.dry_run, args.remove_blurry)
        if result:
            results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Condition':<30} {'Scanned':>8} {'Dup Removed':>12} {'Blurry':>8} {'Remaining':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*12} {'-'*8} {'-'*10}")
    total_removed = 0
    for r in results:
        rd = r['removed_dup']
        total_removed += rd if isinstance(rd, int) else 0
        print(f"  {r['condition']:<30} {r['scanned']:>8} {str(rd):>12} {r['blurry_found']:>8} {r['remaining']:>10}")
    print(f"\n  Total near-duplicates removed: {total_removed}")
    if not args.dry_run and total_removed > 0:
        print(f"  Dataset is now deduplicated. Ready for annotation.")
    elif args.dry_run:
        print(f"  Re-run without --dry-run to apply deletions.")


if __name__ == "__main__":
    main()
