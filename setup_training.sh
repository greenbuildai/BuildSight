#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
#  BuildSight – SASTRA Supercomputer Training Setup Script
#  Run this on node1 to prepare the Conda environment and install deps
# ─────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   BuildSight - Training Environment Setup       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# 1. Activate Conda
echo "Step 1/5: Activating Conda..."
source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate
echo "  ✅ Conda activated"

# 2. Check if 'buildsight' env exists, create if not
echo ""
echo "Step 2/5: Checking for 'buildsight' Conda environment..."
if conda env list | grep -q "buildsight"; then
    echo "  ✅ 'buildsight' environment already exists."
else
    echo "  🔧 Creating 'buildsight' environment (Python 3.10)..."
    conda create -n buildsight python=3.10 -y
    echo "  ✅ Environment created!"
fi

# 3. Activate the buildsight env
echo ""
echo "Step 3/5: Activating 'buildsight' environment..."
conda activate buildsight
echo "  ✅ Activated!"

# 4. Install Ultralytics (YOLO) and PyTorch with CUDA 12.4
echo ""
echo "Step 4/5: Installing Ultralytics + PyTorch (CUDA 12.4)..."
pip install ultralytics --quiet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
echo "  ✅ Dependencies installed!"

# 5. Verify GPU is accessible from within Python
echo ""
echo "Step 5/5: Verifying GPU access from PyTorch..."
python -c "
import torch
print(f'  PyTorch version   : {torch.__version__}')
print(f'  CUDA available    : {torch.cuda.is_available()}')
print(f'  GPU device count  : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU name          : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM              : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ SETUP COMPLETE! Ready for training on A100!"
echo "═══════════════════════════════════════════════════"
echo ""
