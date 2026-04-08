import os
from pathlib import Path

# Paths
DATA_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")
OUTPUT_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset")

src_dir = DATA_DIR / "Normal_Site_Condition"
lbl_root = OUTPUT_DIR / "labels"
seg_root = OUTPUT_DIR / "labels_seg"

if not src_dir.exists():
    print(f"Error: {src_dir} not found")
    exit(1)

files = [f.stem for f in src_dir.glob("*.jpg")]
print(f"Found {len(files)} normal images.")

count = 0
for split in ["train", "val", "test"]:
    for f in files:
        txt = lbl_root / split / (f + ".txt")
        if txt.exists():
            txt.unlink()
            count += 1
        seg = seg_root / split / (f + ".txt")
        if seg.exists():
            seg.unlink()

print(f"Deleted {count} labels for Normal_Site_Condition. Ready for re-run.")
