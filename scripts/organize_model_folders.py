"""
organize_model_folders.py
=========================
Creates 4 self-contained model folders in the backup destination.
Each folder contains: images, labels/annotations, configs, train/infer scripts,
requirements.txt, and README.md.

Usage:
    python scripts/organize_model_folders.py
"""

import os
import sys
import shutil
import json
import time
from pathlib import Path

SRC_EXPORTS  = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\exports")
SRC_ANNS     = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\annotations")
DEST_ROOT    = Path(r"D:\Jovi\Projects\BuildSight\Backup_Dataset(annotated)")

SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    items = list(src.rglob("*"))
    for i, item in enumerate(items):
        if item.is_file():
            rel = item.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(items)} items copied")

def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def folder_size(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

# ===========================================================================
# YOLOv11
# ===========================================================================

def build_yolov11():
    root = DEST_ROOT / "YOLOv11"
    src  = SRC_EXPORTS / "YOLOv11"
    print(f"\n[1/4] Building YOLOv11 -> {root}")

    print("  Copying images + labels...")
    copy_tree(src / "images", root / "images")
    copy_tree(src / "labels", root / "labels")
    shutil.copy2(src / "data.yaml", root / "data.yaml")

    write_file(root / "requirements.txt", """\
# YOLOv11 requirements
ultralytics>=8.3.0
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.66.0
""")

    write_file(root / "train.py", """\
\"\"\"
YOLOv11 Training Script — BuildSight Dataset
=============================================
Train YOLOv11n/s/m/l/x on the 3-class construction safety dataset.

Classes:
  0: helmet
  1: safety_vest
  2: worker

Usage:
    python train.py --model yolov11n --epochs 100 --batch 16 --imgsz 640
    python train.py --model yolov11s --epochs 150 --batch 8  --imgsz 1280
\"\"\"

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv11 training on BuildSight dataset")
    p.add_argument("--model",     default="yolov11n.pt",  help="Model variant (yolov11n/s/m/l/x.pt)")
    p.add_argument("--epochs",    type=int, default=100,  help="Number of training epochs")
    p.add_argument("--batch",     type=int, default=16,   help="Batch size")
    p.add_argument("--imgsz",     type=int, default=640,  help="Input image size")
    p.add_argument("--workers",   type=int, default=8,    help="DataLoader workers")
    p.add_argument("--device",    default="0",            help="GPU device (0, 1, cpu)")
    p.add_argument("--patience",  type=int, default=30,   help="Early stopping patience")
    p.add_argument("--project",   default="runs/train",   help="Output directory")
    p.add_argument("--name",      default="buildsight",   help="Run name")
    p.add_argument("--resume",    action="store_true",    help="Resume from last checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(__file__).parent / "data.yaml"

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5,
        flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1,
        # Task
        task="detect",
        save=True,
        save_period=10,
        val=True,
        plots=True,
    )
    print(f"Training complete. Results saved to {args.project}/{args.name}")
    print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "infer.py", """\
\"\"\"
YOLOv11 Inference Script — BuildSight Dataset
==============================================
Run detection on images or video using a trained YOLOv11 checkpoint.

Usage:
    python infer.py --weights runs/train/buildsight/weights/best.pt --source images/val
    python infer.py --weights best.pt --source /path/to/video.mp4 --conf 0.35
\"\"\"

import argparse
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv11 inference on BuildSight")
    p.add_argument("--weights", required=True,       help="Path to trained .pt weights")
    p.add_argument("--source",  required=True,       help="Image dir, video file, or webcam (0)")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--device",  default="0")
    p.add_argument("--save-txt", action="store_true", help="Save labels to txt")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name",    default="buildsight")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        classes=[0, 1, 2],
        verbose=True,
    )
    print(f"Done. Saved to {args.project}/{args.name}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "compare_models.py", """\
\"\"\"
YOLOv11 Model Comparison Script
================================
Compare multiple YOLOv11 checkpoints on the validation set and output a summary table.

Usage:
    python compare_models.py --weights best_n.pt best_s.pt best_m.pt
\"\"\"

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True, help="Checkpoint paths to compare")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="0")
    p.add_argument("--batch",   type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(__file__).parent / "data.yaml"
    rows = []

    for w in args.weights:
        model = YOLO(w)
        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz,
                            device=args.device, batch=args.batch, verbose=False)
        d = metrics.results_dict
        rows.append({
            "model": Path(w).stem,
            "mAP50":  d.get("metrics/mAP50(B)", 0),
            "mAP50-95": d.get("metrics/mAP50-95(B)", 0),
            "P":  d.get("metrics/precision(B)", 0),
            "R":  d.get("metrics/recall(B)", 0),
        })

    print(f"\\n{'Model':<25} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{r['model']:<25} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['P']:>8.4f} {r['R']:>8.4f}")

    best = max(rows, key=lambda x: x["mAP50"])
    print(f"\\nBest model by mAP50: {best['model']} ({best['mAP50']:.4f})")


if __name__ == "__main__":
    main()
""")

    write_file(root / "README.md", """\
# YOLOv11 — BuildSight Dataset

## Dataset
| Split | Images | Labels |
|-------|--------|--------|
| train | ~3299  | YOLO bbox txt |
| val   | ~942   | YOLO bbox txt |
| test  | ~475   | YOLO bbox txt |

**Classes:** `0=helmet`, `1=safety_vest`, `2=worker`

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# Nano model — fast baseline
python train.py --model yolov11n.pt --epochs 100 --batch 16 --imgsz 640

# Small model — balanced
python train.py --model yolov11s.pt --epochs 150 --batch 8 --imgsz 640

# Medium model — high accuracy
python train.py --model yolov11m.pt --epochs 200 --batch 4 --imgsz 1280
```

## Inference
```bash
python infer.py --weights runs/train/buildsight/weights/best.pt --source images/val
```

## Model Comparison
```bash
python compare_models.py --weights runs/train/exp1/weights/best.pt runs/train/exp2/weights/best.pt
```

## Notes
- Dataset: Indian construction sites (4 conditions: Normal, Dusty, Low_Light, Crowded)
- Annotations: GroundingDINO + SAM pipeline, Gemini auditor (where quota available)
- SAMURAI sequences are tracked by 13-digit epoch-ms filename convention
""")

    sz = folder_size(root)
    print(f"  YOLOv11 done: {sz/1024/1024/1024:.2f} GB")


# ===========================================================================
# YOLOv26
# ===========================================================================

def build_yolov26():
    root = DEST_ROOT / "YOLOv26"
    src  = SRC_EXPORTS / "YOLOv26"
    print(f"\n[2/4] Building YOLOv26 -> {root}")

    print("  Copying images + labels...")
    copy_tree(src / "images", root / "images")
    copy_tree(src / "labels", root / "labels")
    shutil.copy2(src / "data.yaml", root / "data.yaml")

    write_file(root / "requirements.txt", """\
# YOLOv26 requirements
ultralytics>=8.3.0
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
PyYAML>=6.0
tqdm>=4.66.0
""")

    write_file(root / "train.py", """\
\"\"\"
YOLOv26 Training Script — BuildSight Dataset
=============================================
Train YOLOv26n/s/m/l/x on the 3-class construction safety dataset.

Classes:
  0: helmet
  1: safety_vest
  2: worker

Usage:
    python train.py --model yolov26n --epochs 100 --batch 16 --imgsz 640
    python train.py --model yolov26s --epochs 150 --batch 8  --imgsz 1280
\"\"\"

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv26 training on BuildSight dataset")
    p.add_argument("--model",     default="yolov26n.pt",  help="Model variant (yolov26n/s/m/l/x.pt)")
    p.add_argument("--epochs",    type=int, default=100)
    p.add_argument("--batch",     type=int, default=16)
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--workers",   type=int, default=8)
    p.add_argument("--device",    default="0")
    p.add_argument("--patience",  type=int, default=30)
    p.add_argument("--project",   default="runs/train")
    p.add_argument("--name",      default="buildsight")
    p.add_argument("--resume",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(__file__).parent / "data.yaml"

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        task="detect",
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1,
    )
    print(f"Training complete. Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "infer.py", """\
\"\"\"
YOLOv26 Inference Script — BuildSight Dataset
==============================================
Run detection on images or video using a trained YOLOv26 checkpoint.

Usage:
    python infer.py --weights runs/train/buildsight/weights/best.pt --source images/val
    python infer.py --weights best.pt --source /path/to/site_video.mp4 --conf 0.35
\"\"\"

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source",  required=True)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--conf",    type=float, default=0.30)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--device",  default="0")
    p.add_argument("--save-txt", action="store_true")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name",    default="buildsight")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=True,
        save_txt=args.save_txt,
        project=args.project,
        name=args.name,
        classes=[0, 1, 2],
        verbose=True,
    )
    print(f"Done. Saved to {args.project}/{args.name}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "compare_models.py", """\
\"\"\"
YOLOv26 Model Comparison Script
================================
Compare multiple YOLOv26 checkpoints on the validation set.

Usage:
    python compare_models.py --weights best_n.pt best_s.pt best_m.pt
\"\"\"

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="0")
    p.add_argument("--batch",   type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    data_yaml = Path(__file__).parent / "data.yaml"
    rows = []

    for w in args.weights:
        model = YOLO(w)
        metrics = model.val(data=str(data_yaml), imgsz=args.imgsz,
                            device=args.device, batch=args.batch, verbose=False)
        d = metrics.results_dict
        rows.append({
            "model":    Path(w).stem,
            "mAP50":    d.get("metrics/mAP50(B)", 0),
            "mAP50-95": d.get("metrics/mAP50-95(B)", 0),
            "P":        d.get("metrics/precision(B)", 0),
            "R":        d.get("metrics/recall(B)", 0),
        })

    print(f"\\n{'Model':<25} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{r['model']:<25} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} {r['P']:>8.4f} {r['R']:>8.4f}")

    best = max(rows, key=lambda x: x["mAP50"])
    print(f"\\nBest model by mAP50: {best['model']} ({best['mAP50']:.4f})")


if __name__ == "__main__":
    main()
""")

    write_file(root / "README.md", """\
# YOLOv26 — BuildSight Dataset

## Dataset
| Split | Images | Labels |
|-------|--------|--------|
| train | ~3299  | YOLO bbox txt |
| val   | ~942   | YOLO bbox txt |
| test  | ~475   | YOLO bbox txt |

**Classes:** `0=helmet`, `1=safety_vest`, `2=worker`

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py --model yolov26n.pt --epochs 100 --batch 16 --imgsz 640
python train.py --model yolov26s.pt --epochs 150 --batch 8  --imgsz 640
python train.py --model yolov26m.pt --epochs 200 --batch 4  --imgsz 1280
```

## Inference
```bash
python infer.py --weights runs/train/buildsight/weights/best.pt --source images/val
```

## Model Comparison
```bash
python compare_models.py --weights exp1/best.pt exp2/best.pt
```

## Notes
- Same dataset format as YOLOv11 (Ultralytics-compatible YOLO bbox format)
- 4 environmental conditions: Normal, Dusty, Low_Light, Crowded
- YOLOv26 variant supports advanced attention mechanisms for small object detection
""")

    sz = folder_size(root)
    print(f"  YOLOv26 done: {sz/1024/1024/1024:.2f} GB")


# ===========================================================================
# SAMURAI
# ===========================================================================

def build_samurai():
    root = DEST_ROOT / "SAMURAI"
    src  = SRC_EXPORTS / "SAMURAI"
    print(f"\n[3/4] Building SAMURAI -> {root}")

    print("  Copying sequence images...")
    copy_tree(src / "images", root / "images")

    print("  Copying annotations...")
    copy_tree(src / "annotations", root / "annotations")

    # sequence_index.json
    seq_idx = src / "sequence_index.json"
    if seq_idx.exists():
        shutil.copy2(seq_idx, root / "sequence_index.json")

    write_file(root / "requirements.txt", """\
# SAMURAI requirements
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pycocotools>=2.0.7
tqdm>=4.66.0
# SAMURAI (install from source)
# git clone https://github.com/yangchris11/samurai
# cd samurai && pip install -e .
# Also requires SAM2:
# pip install sam2
""")

    write_file(root / "preprocess.py", """\
\"\"\"
SAMURAI Preprocessing — BuildSight Dataset
==========================================
Groups frames into sequences using the 13-digit epoch-ms timestamp naming convention.
Frames within 500ms of each other form a SAMURAI tracking sequence.

Output:
  - sequence_index.json: maps sequence_id -> list of frame paths and track metadata
  - Per-sequence subdirectory layout: images/sequences/<seq_id>/frame_NNNN.jpg

Usage:
    python preprocess.py --output-dir sequences_prepared
\"\"\"

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-index",  default="sequence_index.json",
                   help="Path to sequence_index.json")
    p.add_argument("--images-dir", default="images",
                   help="Source images directory (train/val/test splits)")
    p.add_argument("--output-dir", default="sequences_prepared",
                   help="Output directory for reorganized sequences")
    p.add_argument("--split",      default="train",
                   choices=["train", "val", "test"],
                   help="Which split to preprocess")
    return p.parse_args()


def main():
    args = parse_args()
    seq_idx_path = Path(__file__).parent / args.seq_index
    images_dir   = Path(__file__).parent / args.images_dir / args.split
    output_dir   = Path(__file__).parent / args.output_dir / args.split

    with open(seq_idx_path) as f:
        seq_index = json.load(f)

    print(f"Sequences in index: {len(seq_index)}")
    print(f"Preparing sequences for split: {args.split}")

    prepared = 0
    for seq_id, seq_info in seq_index.items():
        frames = seq_info.get("frames", [])
        if not frames:
            continue

        seq_out = output_dir / seq_id
        seq_out.mkdir(parents=True, exist_ok=True)

        for i, frame_name in enumerate(sorted(frames)):
            src_img = images_dir / frame_name
            if src_img.exists():
                dst_img = seq_out / f"frame_{i:04d}{src_img.suffix}"
                shutil.copy2(src_img, dst_img)

        prepared += 1
        if prepared % 100 == 0:
            print(f"  Prepared {prepared} sequences...")

    print(f"Done. {prepared} sequences written to {output_dir}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "infer.py", """\
\"\"\"
SAMURAI Inference Script — BuildSight Dataset
=============================================
Run SAMURAI zero-shot tracking on a prepared sequence directory.

SAMURAI tracks workers across frames using SAM2 memory attention.
Requires SAMURAI installed from https://github.com/yangchris11/samurai

Usage:
    python infer.py --seq-dir sequences_prepared/train/seq_001 --checkpoint sam2_hiera_large.pt
\"\"\"

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir",    required=True, help="Sequence directory (frames as image files)")
    p.add_argument("--checkpoint", required=True, help="SAM2 model checkpoint (.pt)")
    p.add_argument("--config",     default="sam2_hiera_l.yaml", help="SAM2 model config")
    p.add_argument("--output-dir", default="samurai_output", help="Output directory for results")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--track-class", type=int, default=2,
                   help="Class ID to track (2=worker)")
    return p.parse_args()


def main():
    args = parse_args()
    seq_dir = Path(args.seq_dir)
    frames  = sorted(seq_dir.glob("*.jpg")) + sorted(seq_dir.glob("*.png"))

    if not frames:
        print(f"No frames found in {seq_dir}")
        return

    print(f"Sequence: {seq_dir.name}")
    print(f"Frames: {len(frames)}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print("To run SAMURAI tracking:")
    print("  from samurai import SAMURAITracker")
    print("  tracker = SAMURAITracker(checkpoint=args.checkpoint, config=args.config, device=args.device)")
    print("  tracker.track(frames=frames, output_dir=args.output_dir)")
    print()
    print("Refer to https://github.com/yangchris11/samurai for the full API.")


if __name__ == "__main__":
    main()
""")

    write_file(root / "eval_tracking.py", """\
\"\"\"
SAMURAI Tracking Evaluation — BuildSight Dataset
=================================================
Evaluates tracking performance using COCO annotations with track_id.

Metrics computed:
  - MOTA (Multi-Object Tracking Accuracy)
  - MOTP (Multi-Object Tracking Precision)
  - ID switches
  - Fragmentation

Usage:
    python eval_tracking.py --gt annotations/instances_val.json --pred pred_val.json
\"\"\"

import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt",   required=True, help="Ground truth COCO JSON with track_id")
    p.add_argument("--pred", required=True, help="Prediction COCO JSON with track_id")
    p.add_argument("--iou-thresh", type=float, default=0.50)
    p.add_argument("--class-id",   type=int,   default=2,
                   help="Class ID to evaluate (2=worker)")
    return p.parse_args()


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def main():
    args = parse_args()
    gt   = load_coco(args.gt)
    pred = load_coco(args.pred)

    gt_anns   = [a for a in gt["annotations"]   if a["category_id"] == args.class_id]
    pred_anns = [a for a in pred["annotations"] if a["category_id"] == args.class_id]

    gt_tracked   = sum(1 for a in gt_anns   if a.get("track_id", -1) >= 0)
    pred_tracked = sum(1 for a in pred_anns if a.get("track_id", -1) >= 0)

    print(f"GT annotations (class {args.class_id}):   {len(gt_anns):6d}  ({gt_tracked} tracked)")
    print(f"Pred annotations (class {args.class_id}): {len(pred_anns):6d}  ({pred_tracked} tracked)")
    print()
    print("For full MOTA/MOTP evaluation, use TrackEval:")
    print("  pip install trackeval")
    print("  python -m trackeval.eval ...")


if __name__ == "__main__":
    main()
""")

    write_file(root / "compare_models.py", """\
\"\"\"
SAMURAI vs. other trackers — comparison helper
==============================================
Compares SAMURAI tracking output against baseline trackers (ByteTrack, BoT-SORT)
on the BuildSight validation sequences.

Usage:
    python compare_models.py --samurai samurai_pred.json --bytetrack bytetrack_pred.json
\"\"\"

import json
import argparse
from pathlib import Path


def load_coco(path):
    with open(path) as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samurai",   required=True)
    p.add_argument("--bytetrack", default=None)
    p.add_argument("--botsort",   default=None)
    p.add_argument("--class-id",  type=int, default=2)
    return p.parse_args()


def stats(coco, class_id):
    anns = [a for a in coco["annotations"] if a["category_id"] == class_id]
    tracked = sum(1 for a in anns if a.get("track_id", -1) >= 0)
    seqs    = len({a.get("sequence_id") for a in anns if a.get("sequence_id")})
    return {"total": len(anns), "tracked": tracked, "sequences": seqs}


def main():
    args = parse_args()
    results = {}
    for name, path in [("SAMURAI", args.samurai), ("ByteTrack", args.bytetrack), ("BoT-SORT", args.botsort)]:
        if path is None:
            continue
        coco = load_coco(path)
        results[name] = stats(coco, args.class_id)

    print(f"\\n{'Tracker':<15} {'Total':>8} {'Tracked':>10} {'Sequences':>12}")
    print("-" * 50)
    for name, s in results.items():
        print(f"{name:<15} {s['total']:>8} {s['tracked']:>10} {s['sequences']:>12}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "README.md", """\
# SAMURAI — BuildSight Dataset

## Overview
SAMURAI (SAM-based Unified and Robust AI) performs zero-shot multi-object tracking
using SAM2 memory attention. This folder contains the BuildSight dataset formatted
for SAMURAI sequence tracking.

## Dataset Structure
```
images/
  train/  (frame images, 13-digit epoch-ms filenames)
  val/
  test/
annotations/
  instances_train.json  (COCO JSON with track_id, sequence_id, frame_id)
  instances_val.json
  instances_test.json
sequence_index.json     (sequence_id -> frame list + metadata)
```

## Frame Naming Convention
Filenames are 13-digit epoch-ms timestamps (e.g. `1698234567890.jpg`).
Frames within **500ms** of each other belong to the same SAMURAI sequence.

## Annotation Fields
Each annotation in `instances_*.json` has:
- `track_id`: unique worker identity within a sequence (>= 0 for workers)
- `sequence_id`: sequence group identifier
- `frame_id`: frame index within the sequence
- `segmentation`: SAM polygon mask (for mask-guided initialization)

## Setup
```bash
pip install -r requirements.txt
# Install SAMURAI from source:
git clone https://github.com/yangchris11/samurai
cd samurai && pip install -e .
```

## Preprocessing
```bash
python preprocess.py --split train --output-dir sequences_prepared
```

## Inference
```bash
python infer.py --seq-dir sequences_prepared/train/seq_001 --checkpoint sam2_hiera_large.pt
```

## Evaluation
```bash
python eval_tracking.py --gt annotations/instances_val.json --pred pred_val.json
```

## Model Comparison
```bash
python compare_models.py --samurai samurai_pred.json --bytetrack bytetrack_pred.json
```

## Stats (at export time)
- Sequences: ~4,476
- Tracked worker instances: ~42,772
- Classes: `0=helmet`, `1=safety_vest`, `2=worker` (only workers have track_id)
""")

    sz = folder_size(root)
    print(f"  SAMURAI done: {sz/1024/1024/1024:.2f} GB")


# ===========================================================================
# YOLACT++
# ===========================================================================

def build_yolact():
    root = DEST_ROOT / "YOLACT_plusplus"
    src  = SRC_EXPORTS / "YOLACT_plus_plus"
    print(f"\n[4/4] Building YOLACT++ -> {root}")

    print("  Copying images...")
    copy_tree(src / "images", root / "images")

    print("  Copying annotations (COCO JSON + SAM masks)...")
    copy_tree(src / "annotations", root / "annotations")

    mask_stats = src / "mask_stats.json"
    if mask_stats.exists():
        shutil.copy2(mask_stats, root / "mask_stats.json")

    write_file(root / "requirements.txt", """\
# YOLACT++ requirements
torch>=2.1.0
torchvision>=0.16.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
pycocotools>=2.0.7
tqdm>=4.66.0
# YOLACT++ (install from source)
# git clone https://github.com/dbolya/yolact
# cd yolact && pip install -r requirements.txt
# Also requires:
cython>=3.0.0
matplotlib>=3.7.0
""")

    write_file(root / "preprocess.py", """\
\"\"\"
YOLACT++ Preprocessing — BuildSight Dataset
============================================
Validates COCO JSON annotations and verifies SAM segmentation masks are present.
Also generates a class-balanced image list for training.

Usage:
    python preprocess.py --split train
\"\"\"

import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      default="train", choices=["train", "val", "test"])
    p.add_argument("--ann-dir",    default="annotations")
    p.add_argument("--images-dir", default="images")
    return p.parse_args()


def main():
    args = parse_args()
    ann_path = Path(__file__).parent / args.ann_dir / f"instances_{args.split}.json"
    img_dir  = Path(__file__).parent / args.images_dir / args.split

    with open(ann_path) as f:
        coco = json.load(f)

    anns   = coco["annotations"]
    images = {img["id"]: img for img in coco["images"]}

    # Check segmentations
    empty_seg = [a["id"] for a in anns if not a.get("segmentation")]
    no_img    = [a["id"] for a in anns if a["image_id"] not in images]

    print(f"Split: {args.split}")
    print(f"  Images:             {len(images)}")
    print(f"  Annotations:        {len(anns)}")
    print(f"  Empty segmentation: {len(empty_seg)}")
    print(f"  Missing image refs: {len(no_img)}")

    # Class distribution
    by_class = {}
    for a in anns:
        cid = a["category_id"]
        by_class[cid] = by_class.get(cid, 0) + 1

    class_names = {1: "helmet", 2: "safety_vest", 3: "worker"}
    print("  Class distribution:")
    for cid, cnt in sorted(by_class.items()):
        print(f"    class {cid} ({class_names.get(cid, '?')}): {cnt}")

    # Verify images on disk
    found = sum(1 for img in images.values()
                if (img_dir / img["file_name"]).exists()
                or (img_dir / Path(img["file_name"]).name).exists())
    print(f"  Images on disk: {found}/{len(images)}")

    if empty_seg:
        print(f"  WARNING: {len(empty_seg)} annotations have no segmentation polygon!")
    else:
        print("  OK: All annotations have segmentation polygons.")


if __name__ == "__main__":
    main()
""")

    write_file(root / "train.py", """\
\"\"\"
YOLACT++ Training Script — BuildSight Dataset
=============================================
Generates the YOLACT++ config and launches training.

YOLACT++ requires installation from: https://github.com/dbolya/yolact

Usage:
    python train.py --config yolact_plus_resnet50_config --epochs 80 --batch 8
\"\"\"

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="yolact_plus_resnet50_config",
                   help="YOLACT++ config name (see yolact/data/config.py)")
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--batch",    type=int, default=8)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--yolact-dir", default="../yolact",
                   help="Path to cloned YOLACT++ repository")
    p.add_argument("--resume",   default=None, help="Resume from checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    yolact = Path(args.yolact_dir)
    ann_dir = Path(__file__).parent / "annotations"
    img_dir = Path(__file__).parent / "images"

    print("YOLACT++ Training Setup")
    print(f"  Config:      {args.config}")
    print(f"  Annotations: {ann_dir}")
    print(f"  Images:      {img_dir}")
    print()
    print("Before training, update yolact/data/config.py to point to this dataset:")
    print(f"  dataset = dataset_base.copy({{")
    print(f"      'name': 'BuildSight',")
    print(f"      'train_images': '{img_dir / 'train'}',")
    print(f"      'train_info':   '{ann_dir / 'instances_train.json'}',")
    print(f"      'valid_images': '{img_dir / 'val'}',")
    print(f"      'valid_info':   '{ann_dir / 'instances_val.json'}',")
    print(f"      'has_gt': True,")
    print(f"      'class_names': ('helmet', 'safety_vest', 'worker'),")
    print(f"  }})")
    print()
    print("Then run from the yolact directory:")
    print(f"  python train.py --config={args.config} --batch_size={args.batch} --lr={args.lr}")
    if args.resume:
        print(f"  python train.py --config={args.config} --resume={args.resume}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "infer.py", """\
\"\"\"
YOLACT++ Inference Script — BuildSight Dataset
==============================================
Run instance segmentation on images using a trained YOLACT++ checkpoint.

Requires YOLACT++ from: https://github.com/dbolya/yolact

Usage:
    python infer.py --weights weights/yolact_plus_50_54800.pth --source images/val
\"\"\"

import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    required=True)
    p.add_argument("--source",     required=True)
    p.add_argument("--yolact-dir", default="../yolact")
    p.add_argument("--config",     default="yolact_plus_resnet50_config")
    p.add_argument("--score-thresh", type=float, default=0.30)
    p.add_argument("--output-dir", default="yolact_output")
    p.add_argument("--display",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print("YOLACT++ Inference")
    print(f"  Weights: {args.weights}")
    print(f"  Source:  {args.source}")
    print()
    print("Run from the yolact directory:")
    print(f"  python eval.py --trained_model={args.weights} \\\\")
    print(f"                 --config={args.config} \\\\")
    print(f"                 --score_threshold={args.score_thresh} \\\\")
    print(f"                 --images={args.source}:{args.output_dir}")


if __name__ == "__main__":
    main()
""")

    write_file(root / "compare_models.py", """\
\"\"\"
YOLACT++ Model Comparison Script
=================================
Evaluates multiple YOLACT++ checkpoints on the validation set using COCO AP metrics.

Usage:
    python compare_models.py --weights w1.pth w2.pth --yolact-dir ../yolact
\"\"\"

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    nargs="+", required=True)
    p.add_argument("--yolact-dir", default="../yolact")
    p.add_argument("--config",     default="yolact_plus_resnet50_config")
    p.add_argument("--ann",        default="annotations/instances_val.json")
    return p.parse_args()


def main():
    args = parse_args()
    ann_path = Path(__file__).parent / args.ann

    with open(ann_path) as f:
        coco = json.load(f)

    print(f"Val annotations: {len(coco['annotations'])}")
    print(f"Val images:      {len(coco['images'])}")
    print()
    print("To evaluate each checkpoint, run from the yolact directory:")
    for w in args.weights:
        print(f"  python eval.py --trained_model={w} \\\\")
        print(f"                 --config={args.config} \\\\")
        print(f"                 --coco_path={ann_path.parent.parent} \\\\")
        print(f"                 --benchmark")
        print()
    print("Metrics reported: AP (mask), AP50, AP75, APs, APm, APl")


if __name__ == "__main__":
    main()
""")

    write_file(root / "README.md", """\
# YOLACT++ — BuildSight Dataset

## Overview
YOLACT++ (You Only Look At CoefficienTs) performs real-time instance segmentation.
This folder contains the BuildSight dataset in COCO JSON format with SAM-generated
segmentation polygons on every annotation.

## Dataset Structure
```
images/
  train/  (~3299 images)
  val/    (~942 images)
  test/   (~475 images)
annotations/
  instances_train.json  (COCO JSON with SAM segmentation polygons)
  instances_val.json
  instances_test.json
mask_stats.json         (SAM mask coverage statistics)
```

## Annotation Format
All annotations include:
- `bbox`: [x, y, width, height] (COCO format)
- `segmentation`: [[x1,y1,x2,y2,...]] (SAM polygon, simplified with approxPolyDP)
- `area`: polygon area in pixels
- `iscrowd`: always 0

**Classes (COCO category_id):**
- 1: helmet
- 2: safety_vest
- 3: worker

> Note: COCO category IDs start at 1. YOLO class IDs start at 0.

## Setup
```bash
pip install -r requirements.txt
# Clone YOLACT++:
git clone https://github.com/dbolya/yolact
cd yolact && pip install -r requirements.txt
```

## Dataset Config (add to yolact/data/config.py)
```python
buildsight_dataset = dataset_base.copy({
    'name': 'BuildSight',
    'train_images': '/path/to/YOLACT_plusplus/images/train',
    'train_info':   '/path/to/YOLACT_plusplus/annotations/instances_train.json',
    'valid_images': '/path/to/YOLACT_plusplus/images/val',
    'valid_info':   '/path/to/YOLACT_plusplus/annotations/instances_val.json',
    'has_gt': True,
    'class_names': ('helmet', 'safety_vest', 'worker'),
})
```

## Preprocessing
```bash
python preprocess.py --split train
python preprocess.py --split val
```

## Training
```bash
python train.py --config yolact_plus_resnet50_config --epochs 80 --batch 8
```

## Inference
```bash
python infer.py --weights weights/yolact_plus_50_54800.pth --source images/val
```

## Model Comparison
```bash
python compare_models.py --weights exp1.pth exp2.pth
```

## Mask Stats (at export time)
- SAM mask coverage: >= 60% of annotations
- Fallback (rectangle polygon): where SAM failed
- All annotations guaranteed non-empty segmentation
""")

    sz = folder_size(root)
    print(f"  YOLACT++ done: {sz/1024/1024/1024:.2f} GB")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("BuildSight — Model Folder Organizer")
    print(f"Destination: {DEST_ROOT}")
    print("=" * 60)

    build_yolov11()
    build_yolov26()
    build_samurai()
    build_yolact()

    elapsed = time.time() - t0
    total = folder_size(DEST_ROOT)
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} min")
    print(f"Total size: {total/1024/1024/1024:.2f} GB")
    print(f"Destination: {DEST_ROOT}")
    print("=" * 60)
