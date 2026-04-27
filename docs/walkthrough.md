# BuildSight – SASTRA Training Walkthrough

This document records the successful preparation and environment setup for training the BuildSight YOLO model on the SASTRA supercomputer.

## 🏁 Phase 1: Dataset Integrity & Transfer
- **Total Files:** 32,984 files verified on SASTRA NFS storage.
- **Total Size:** 11.75 GB (Verified via `du -sh`).
- **Audit:** 100% data integrity confirmed between local source and remote target.

## 🛠️ Phase 2: Environment Setup (Node1)
- **Conda Environment:** `buildsight` (Python 3.10) created.
- **Library Installation:** Successfully installed `ultralytics`, `torch` (with CUDA 12 support), and `nvidia-cuda` libraries.
- **Background Mode:** Final installation was handled via `nohup` to ensure persistence during terminal disconnects.

## 🚀 Phase 3: Training Launch Configuration
- **Compute:** A100 40GB GPU (Node1).
- **Model:** YOLOv11n (Nano).
- **Epochs:** 100.
- **Wait Status:** Ready for execution.

## 🎬 Final Step
To begin training, the following command must be executed on `node1`:
```bash
conda activate buildsight
python ~/train_buildsight.py
```
