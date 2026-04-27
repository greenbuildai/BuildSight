# BuildSight Annotation Pipeline — Setup & Execution Guide

## Current Dataset Status (post-QA)

| Condition | Final Images | Notes |
|---|---|---|
| Normal_Site_Condition | 1,373 | Cleaned |
| Crowded_Condition | 991 | Cleaned + 284 video frames added |
| Dusty_Condition | 1,261 | Cleaned |
| Low_Light_Condition | 1,087 | Cleaned |
| **TOTAL** | **4,712** | Ready for annotation |

**Removed:** ~660 near-duplicate images (pHash dedup, Hamming <= 8)
**Retained:** All blurry images in Dusty/Low-Light (intentional visual degradation = training signal)

---

## Pre-Annotation QA — DONE

- [x] GCP credentials files deleted from dataset folder AND project root
- [x] audit_dataset.py run — 0 corrupted, 0 low-res, 0 exact duplicates
- [x] extract_frames.py run — 284 frames extracted from 2 MP4 videos
- [x] pHash dedup run — 660 near-duplicates removed across all conditions
- [x] data.yaml created at Dataset/Final_Annotated_Dataset/data.yaml
- [x] annotate_indian_dataset.py built (full 4-class pipeline)
- [x] export_formats.py fixed (class_id schema corrected)

---

## Step 1: Install Full Annotation Dependencies

```bash
# CUDA 12.1 (match your GPU driver version)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Core annotation packages
pip install opencv-python numpy Pillow tqdm imagehash

# Segment Anything Model (SAM)
pip install git+https://github.com/facebookresearch/segment-anything.git

# GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

---

## Step 2: Download Model Weights

Create the weights folder:
```bash
mkdir -p "E:\Company\Green Build AI\Prototypes\BuildSight\weights"
cd "E:\Company\Green Build AI\Prototypes\BuildSight\weights"
```

Download SAM ViT-H checkpoint (~2.4 GB):
```bash
# Option A: wget
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Option B: Python
python -c "
import urllib.request
url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
urllib.request.urlretrieve(url, 'sam_vit_h_4b8939.pth')
print('SAM downloaded')
"
```

Download GroundingDINO SwinT weights (~700 MB):
```bash
# Option A: wget
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Option B: Python
python -c "
import urllib.request
url = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
urllib.request.urlretrieve(url, 'groundingdino_swint_ogc.pth')
print('GroundingDINO downloaded')
"
```

Clone GroundingDINO repo (needed for config file):
```bash
cd "E:\Company\Green Build AI\Prototypes\BuildSight"
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

Expected weights folder structure:
```
weights/
  sam_vit_h_4b8939.pth          (~2.4 GB)
  groundingdino_swint_ogc.pth   (~700 MB)
GroundingDINO/
  groundingdino/
    config/
      GroundingDINO_SwinT_OGC.py
```

---

## Step 3: Verify Setup

```bash
cd "E:\Company\Green Build AI\Prototypes\BuildSight"
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

from segment_anything import sam_model_registry
print("SAM import OK")

from groundingdino.util.inference import load_model
print("GroundingDINO import OK")

import os
weights_dir = r"E:\Company\Green Build AI\Prototypes\BuildSight\weights"
for f in ["sam_vit_h_4b8939.pth", "groundingdino_swint_ogc.pth"]:
    path = os.path.join(weights_dir, f)
    print(f"  {f}: {'EXISTS' if os.path.exists(path) else 'MISSING'}")
EOF
```

---

## Step 4: Run Annotation Pipeline

### Smoke test (50 images per condition, bbox only — fast, no GPU RAM pressure):
```bash
cd "E:\Company\Green Build AI\Prototypes\BuildSight"
python scripts/annotate_indian_dataset.py --test-batch 50 --skip-sam
```

### Single condition test (Normal only, with SAM masks):
```bash
python scripts/annotate_indian_dataset.py --conditions Normal_Site_Condition
```

### Full pipeline — ALL conditions (requires ~8 GB VRAM for SAM ViT-H):
```bash
python scripts/annotate_indian_dataset.py
```

### Full pipeline — bbox only (no SAM, ~1 GB VRAM, much faster):
```bash
python scripts/annotate_indian_dataset.py --skip-sam
```

Output will appear at:
```
Dataset/Final_Annotated_Dataset/
  images/train|val|test/
  labels/train|val|test/        <- YOLO bbox .txt
  labels_seg/train|val|test/    <- YOLO polygon .txt
  annotations/
    instances_train.json        <- COCO with segmentation + attributes
    instances_val.json
    instances_test.json
  data.yaml
```

---

## Step 5: Human Review (LabelStudio)

Install LabelStudio:
```bash
pip install label-studio
label-studio start
```

Import the generated COCO JSON:
1. Open LabelStudio at http://localhost:8080
2. Create project → Image Object Detection
3. Import → instances_train.json (COCO format)
4. Review box tightness, missing annotations, attribute flags
5. For Dusty_Condition: draw dust_zone polygons
6. Export back as COCO JSON when done

---

## Step 6: Re-export YOLO from Reviewed COCO

After human review, regenerate YOLO .txt files from the corrected COCO JSON:
```bash
EXPORT_SPLIT=train python scripts/export_formats.py
EXPORT_SPLIT=val   python scripts/export_formats.py
EXPORT_SPLIT=test  python scripts/export_formats.py
```

---

## Step 7: QA Spot-Check

```bash
python scripts/visualize_annotations.py
```

---

## Class Schema Reference (LOCKED)

| class_id | name         | COCO category_id |
|----------|--------------|-----------------|
| 0        | helmet       | 0               |
| 1        | safety_vest  | 1               |
| 2        | reserved     | —               |
| 3        | safety_boots | 3               |
| 4        | reserved     | —               |
| 5        | reserved     | —               |
| 6        | worker       | 6               |

**Never change these IDs.** All downstream models (YOLOv11, YOLOv26, SAMURAI, YOLACT++) use this schema directly.

---

## GPU Requirements

| Mode | VRAM Required | Speed (RTX 4050) |
|---|---|---|
| GroundingDINO only (--skip-sam) | ~2 GB | ~15 img/sec |
| GroundingDINO + SAM ViT-H | ~8 GB | ~2 img/sec |
| GroundingDINO + SAM ViT-B (lighter) | ~4 GB | ~5 img/sec |

For RTX 4050 Notebook (6 GB VRAM): use `--skip-sam` for first pass, add masks in LabelStudio manually for difficult cases.

---

## Security Reminder

Two GCP service account key files have been deleted:
- `Dataset/Indian Dataset/ecocraft-designer-470803-96ef945c1dc0.json` (deleted)
- `ecocraft-designer-470803-96ef945c1dc0.json` (root, deleted)

**You must revoke this key at GCP Console → IAM → Service Accounts → Manage Keys.**
Deleting the file does NOT invalidate the key — only GCP console revocation does.
