"""
BuildSight – YOLO Model Training Script
Runs on SASTRA Supercomputer (A100 40GB GPU)
"""
from ultralytics import YOLO
import torch
import os

# ── Config ───────────────────────────────────────────────────────────
MODEL       = "yolo11n.pt"           # Starting checkpoint (nano for fast iteration)
DATASET_DIR = "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11"
YAML_PATH   = f"{DATASET_DIR}/data.yaml"
RUN_NAME    = "buildsight_v1"
EPOCHS      = 100
BATCH_SIZE  = 16                     # Safe for A100 40GB
IMG_SIZE    = 640
DEVICE      = 0                      # GPU 0 (A100)
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  BuildSight Model Training – SASTRA Supercomputer")
    print("═"*60)
    print(f"  Model     : {MODEL}")
    print(f"  Dataset   : {DATASET_DIR}")
    print(f"  Epochs    : {EPOCHS}")
    print(f"  Batch     : {BATCH_SIZE}")
    print(f"  Device    : {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("═"*60 + "\n")

    if not os.path.exists(YAML_PATH):
        print(f"  ⚠️  WARNING: {YAML_PATH} not found!")
        print("  Please ensure data.yaml exists in the YOLOv11 dataset folder.")
        return

    # Load the model
    model = YOLO(MODEL)

    # Train
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        name=RUN_NAME,
        workers=4,
        cache=True,
        verbose=True,
    )

    print("\n  ✅ Training complete!")
    print(f"  Results saved to: runs/detect/{RUN_NAME}")

if __name__ == "__main__":
    main()
