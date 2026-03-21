"""Verify that every image from Dataset_3 and Dataset_4 exists in a condition folder."""
from pathlib import Path

BASE = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".mpo"}

# Source datasets
sources = ["PPE_SASTRA_Dataset_3", "PPE_SASTRA_Dataset_4"]

# Destination folders
dest_folders = ["Normal_Site_Condition", "Dusty_Condition", "Low_Light_Condition", "Crowded_Condition"]

# Build a set of ALL filenames in destination folders
dest_files = {}
for folder_name in dest_folders:
    folder = BASE / folder_name
    if folder.exists():
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXT:
                dest_files[f.name] = folder_name

print(f"Total files across all 4 condition folders: {len(dest_files)}")
print()

# Now check each source dataset
total_source = 0
total_found = 0
total_missing = 0
missing_files = []

for dataset in sources:
    src = BASE / dataset
    if not src.exists():
        print(f"  [WARN] {dataset} folder not found!")
        continue
    
    src_files = [f for f in src.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXT]
    found = 0
    missing = 0
    
    for f in src_files:
        if f.name in dest_files:
            found += 1
        else:
            # Also check with dataset suffix (collision rename)
            stem = f.stem
            suffix = f.suffix
            alt_name = f"{stem}_{dataset}{suffix}"
            if alt_name in dest_files:
                found += 1
            else:
                missing += 1
                missing_files.append(f.name)
    
    total_source += len(src_files)
    total_found += found
    total_missing += missing
    
    status = "✅ ALL MATCHED" if missing == 0 else f"❌ {missing} MISSING"
    print(f"  {dataset}: {len(src_files)} images | Found in dest: {found} | Missing: {missing} | {status}")

print()
print(f"{'='*60}")
print(f"  GRAND TOTAL")
print(f"  Source images:     {total_source}")
print(f"  Matched in dest:   {total_found}")
print(f"  MISSING:           {total_missing}")
print(f"{'='*60}")

if missing_files:
    print(f"\n  First 20 missing files:")
    for f in missing_files[:20]:
        print(f"    - {f}")

if total_missing == 0:
    print("\n  ✅ VERIFICATION PASSED: Every image is categorized!")
else:
    print(f"\n  ❌ VERIFICATION FAILED: {total_missing} images are NOT categorized yet.")
