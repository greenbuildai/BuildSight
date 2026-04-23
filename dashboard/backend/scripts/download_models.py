"""
BuildSight — Model Download Script
====================================
Downloads intelligence model weights:
  1. SAM (vit_b) for segmentation
  2. Florence-2-base for VLM narration (from HuggingFace Hub)

Note: Florence-2-base will auto-download from HF cache on first use
via geoai_vlm_util.py. This script pre-downloads it for offline use.
"""

import os
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
SAM_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_b.pth")

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
VLM_REPO = "microsoft/Florence-2-base"

def download_file(url, path):
    print(f"Downloading {url} to {path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(path, 'wb') as file, tqdm(
        desc=os.path.basename(path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def main():
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR)
        print(f"Created {WEIGHTS_DIR}")

    # 1. Download SAM
    if not os.path.exists(SAM_PATH):
        download_file(SAM_URL, SAM_PATH)
    else:
        print(f"SAM already exists at {SAM_PATH}")

    # 2. Pre-download Florence-2-base to HF cache
    # Florence-2 will be loaded from HF cache by geoai_vlm_util.py at runtime.
    # This just ensures the weights are cached for offline/air-gapped use.
    print(f"Pre-downloading Florence-2-base ({VLM_REPO}) to HuggingFace cache...")
    try:
        snapshot_download(repo_id=VLM_REPO, revision="main")
        print("Florence-2-base cached successfully.")
    except Exception as e:
        print(f"Warning: Could not pre-download Florence-2-base: {e}")
        print("The model will be downloaded automatically on first use.")

    print("\n[SUCCESS] Intelligence models downloaded.")

if __name__ == "__main__":
    main()
