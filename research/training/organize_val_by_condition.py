"""
organize_val_by_condition.py
============================
Classifies val-split annotated images by site condition using filename keywords,
then copies them into condition subfolders for all 4 models.

Output structure:
  /nfsshare/joseva/val_annotated_by_condition/
    S1_normal/
      YOLOv11/
      YOLOv26/
      YOLACT_plusplus/
      SAMURAI_GT/
    S2_dusty/
      ...
    S3_low_light/
      ...
    S4_crowded/
      ...
    unclassified/       <- any images that don't match a keyword
      ...

Condition classification rules (applied to lowercase filename):
  S1 normal    : 'normal'
  S2 dusty     : 'dusty'
  S3 low_light : 'low_light', 'lowlight', 'low-light', 'night', 'dark'
  S4 crowded   : 'crowded', 'crowd'

Run on SASTRA node1:
  /nfsshare/joseva/.conda/envs/buildsight/bin/python \\
      /nfsshare/joseva/organize_val_by_condition.py

Or locally (adjust ANNOTATED_ROOT / ANN_JSON):
  python scripts/organize_val_by_condition.py
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config — adjust if running locally vs on SASTRA
# ---------------------------------------------------------------------------

ANNOTATED_ROOT = Path('/nfsshare/joseva/val_annotated')
ANN_JSON       = Path('/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_val.json')
OUT_ROOT       = Path('/nfsshare/joseva/val_annotated_by_condition')

MODELS = ['YOLOv11', 'YOLOv26', 'YOLACT_plusplus', 'SAMURAI_GT']

# Keyword → condition folder name (checked in order; first match wins)
CONDITION_RULES = [
    (['dusty'],                           'S2_dusty'),
    (['low_light', 'lowlight', 'low-light', 'night', 'dark'], 'S3_low_light'),
    (['crowded', 'crowd'],                'S4_crowded'),
    (['normal'],                          'S1_normal'),
]

# ---------------------------------------------------------------------------
# Step 1 — Load val filenames from COCO JSON
# ---------------------------------------------------------------------------

print(f'Loading val annotation JSON: {ANN_JSON}')
with open(ANN_JSON) as f:
    coco = json.load(f)

images = coco['images']
print(f'Total val images in JSON: {len(images)}')

# ---------------------------------------------------------------------------
# Step 2 — Classify each filename
# ---------------------------------------------------------------------------

def classify(filename: str) -> str:
    fn = filename.lower()
    for keywords, condition in CONDITION_RULES:
        if any(kw in fn for kw in keywords):
            return condition
    return 'unclassified'

condition_map = {}   # basename → condition
counts = defaultdict(int)

for img in images:
    fname = os.path.basename(img['file_name'])
    cond  = classify(fname)
    condition_map[fname] = cond
    counts[cond] += 1

print('\nCondition distribution from filenames:')
for cond, n in sorted(counts.items()):
    print(f'  {cond}: {n} images')

unclassified_n = counts.get('unclassified', 0)
if unclassified_n > 0:
    print(f'\nWARNING: {unclassified_n} images could not be classified by filename.')
    print('  Consider adding more keywords to CONDITION_RULES or using a classifier.')

# ---------------------------------------------------------------------------
# Step 3 — Copy annotated images into condition subfolders
# ---------------------------------------------------------------------------

# Create all output dirs
all_conditions = set(condition_map.values())
for cond in all_conditions:
    for model in MODELS:
        (OUT_ROOT / cond / model).mkdir(parents=True, exist_ok=True)

copied   = defaultdict(lambda: defaultdict(int))  # model → condition → count
missing  = defaultdict(list)                        # model → [missing basenames]

for model in MODELS:
    src_dir = ANNOTATED_ROOT / model
    if not src_dir.exists():
        print(f'\nWARNING: source folder missing — {src_dir}')
        continue

    src_files = {f.name: f for f in src_dir.iterdir() if f.suffix in ('.jpg', '.jpeg', '.png')}
    print(f'\n{model}: {len(src_files)} annotated images found in source')

    for fname, cond in condition_map.items():
        if fname in src_files:
            dst = OUT_ROOT / cond / model / fname
            if not dst.exists():
                shutil.copy2(src_files[fname], dst)
            copied[model][cond] += 1
        else:
            missing[model].append(fname)

# ---------------------------------------------------------------------------
# Step 4 — Report
# ---------------------------------------------------------------------------

print('\n' + '='*60)
print('COPY SUMMARY')
print('='*60)

total_copied = 0
for model in MODELS:
    print(f'\n{model}:')
    model_total = 0
    for cond in sorted(copied[model]):
        n = copied[model][cond]
        print(f'  {cond}: {n}')
        model_total += n
    total_copied += model_total
    if missing[model]:
        print(f'  MISSING (not yet annotated): {len(missing[model])} images')
    print(f'  TOTAL: {model_total}')

print(f'\nGrand total images copied: {total_copied}')
print(f'Output root: {OUT_ROOT}')

# ---------------------------------------------------------------------------
# Step 5 — Write a summary JSON
# ---------------------------------------------------------------------------

summary = {
    'condition_counts_from_filenames': dict(counts),
    'copy_results': {
        model: {cond: n for cond, n in copied[model].items()}
        for model in MODELS
    },
    'missing_per_model': {
        model: len(missing[model]) for model in MODELS
    }
}

out_json = OUT_ROOT / 'val_condition_summary.json'
with open(out_json, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nSummary written to: {out_json}')
print('Done.')
