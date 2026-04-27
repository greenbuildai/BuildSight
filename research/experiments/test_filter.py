import json
import os
from pathlib import Path

common_set = set()
p = Path("/nfsshare/joseva/common_888.txt")
if p.exists():
    with open(p) as f:
        common_set = {line.strip() for line in f if line.strip()}
print(f"Loaded {len(common_set)} common images from {p}")

ann_path = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/SAMURAI/annotations/instances_val.json")
if not ann_path.exists():
    print(f"ERROR: {ann_path} not found")
    exit(1)

with open(ann_path) as f:
    coco = json.load(f)

print(f"Total images in JSON: {len(coco['images'])}")

valid_images = [img for img in coco["images"] if (not common_set or img["file_name"] in common_set)]
print(f"Filtered to {len(valid_images)} images.")

if len(coco['images']) > 0:
    print(f"Sample file_name from JSON: '{coco['images'][0]['file_name']}'")
if len(common_set) > 0:
    print(f"Sample from common_set: '{list(common_set)[0]}'")
