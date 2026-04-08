import os
from pathlib import Path

common_888 = set()
p_common = Path("/nfsshare/joseva/common_888.txt")
if not p_common.exists():
    print(f"ERROR: {p_common} not found")
    exit(1)

with open(p_common) as f:
    common_888 = {line.strip() for line in f if line.strip()}
print(f"Loaded {len(common_888)} common image names.")

folders = [
    "/nfsshare/joseva/val_annotated/YOLOv11",
    "/nfsshare/joseva/val_annotated/YOLOv26",
    "/nfsshare/joseva/val_annotated/YOLACT_plusplus",
    "/nfsshare/joseva/val_annotated/SAMURAI_GT"
]

for folder in folders:
    p = Path(folder)
    if not p.exists():
        print(f"Skipping missing: {folder}")
        continue
    
    files = list(p.glob("*.jpg"))
    print(f"Checking {folder}: {len(files)} files found.")
    
    deleted = 0
    for f in files:
        if f.name not in common_888:
            f.unlink()
            deleted += 1
            
    current_count = len(list(p.glob("*.jpg")))
    print(f"Deleted {deleted} extra files in {folder}. Final count: {current_count}")
