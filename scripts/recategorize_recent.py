"""
Re-categorize Recently Processed Images with API Verification
==============================================================
Moves recently categorized images back to source and re-runs with API.
"""

import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta

# Configuration
BASE_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")

DEST_FOLDERS = {
    "Normal_Site_Condition": BASE_DIR / "Normal_Site_Condition",
    "Dusty_Condition": BASE_DIR / "Dusty_Condition",
    "Low_Light_Condition": BASE_DIR / "Low_Light_Condition",
    "Crowded_Condition": BASE_DIR / "Crowded_Condition",
}

SOURCE_DATASETS = [
    BASE_DIR / "PPE_SASTRA_Dataset_3",
    BASE_DIR / "PPE_SASTRA_Dataset_4",
]

# Find images modified in last N minutes
RECENT_MINUTES = 10

def find_recent_images(minutes=10):
    """Find images modified in the last N minutes"""
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent_images = []

    for category, folder in DEST_FOLDERS.items():
        if not folder.exists():
            continue

        for img_path in folder.iterdir():
            if not img_path.is_file():
                continue

            # Check modification time
            mtime = datetime.fromtimestamp(img_path.stat().st_mtime)

            if mtime > cutoff_time:
                recent_images.append({
                    'path': img_path,
                    'category': category,
                    'mtime': mtime
                })

    return recent_images

def move_back_to_source(recent_images):
    """Move images back to source datasets"""
    moved_count = 0

    # Ensure source folders exist
    for src in SOURCE_DATASETS:
        src.mkdir(parents=True, exist_ok=True)

    # Simple strategy: alternate between datasets to distribute evenly
    dataset_idx = 0

    for img_info in recent_images:
        src_path = img_info['path']

        # Choose destination dataset (alternate)
        dest_dataset = SOURCE_DATASETS[dataset_idx % len(SOURCE_DATASETS)]
        dest_path = dest_dataset / src_path.name

        # Handle collision
        if dest_path.exists():
            base = src_path.stem
            suffix = src_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dataset / f"{base}_{counter}{suffix}"
                counter += 1

        # Move file
        shutil.move(str(src_path), str(dest_path))
        moved_count += 1
        dataset_idx += 1

    return moved_count

def main():
    print("=" * 70)
    print("Re-categorize Recent Images with API Verification")
    print("=" * 70)

    # Step 1: Find recent images
    print(f"\n[Step 1] Finding images modified in last {RECENT_MINUTES} minutes...")
    recent_images = find_recent_images(RECENT_MINUTES)

    print(f"Found {len(recent_images)} recently categorized images:")

    # Count by category
    by_category = {}
    for img in recent_images:
        cat = img['category']
        by_category[cat] = by_category.get(cat, 0) + 1

    for cat, count in by_category.items():
        print(f"  {cat}: {count} images")

    if not recent_images:
        print("\nNo recent images found. Nothing to re-categorize.")
        return

    # Step 2: Confirm
    print(f"\n[Step 2] Moving {len(recent_images)} images back to source datasets...")

    # Move back
    moved = move_back_to_source(recent_images)
    print(f"✓ Moved {moved} images back to source")

    # Step 3: Run hybrid classification
    print("\n[Step 3] Running hybrid classification with API verification...")
    print("=" * 70)

    # Set API keys
    os.environ['GEMINI_KEY_1'] = 'AIzaSyAuHrM7wY5T3r7HYDIBPc9us0ai-lK76_E'
    os.environ['GEMINI_KEY_2'] = 'AIzaSyDMq7A5DZDKM4hJbLNeS2-qpTsNJ7HCmiA'

    # Run hybrid script
    import subprocess
    result = subprocess.run([
        sys.executable,
        'categorize_hybrid.py',
        '--batch-size', '48',
        '--confidence-threshold', '0.75'
    ])

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("✓ Re-categorization complete!")
    else:
        print("⚠ Re-categorization completed with warnings")
    print("=" * 70)

if __name__ == "__main__":
    main()
