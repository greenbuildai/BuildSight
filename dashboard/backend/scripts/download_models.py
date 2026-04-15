import os
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
SAM_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_b.pth")
VLM_PATH = os.path.join(WEIGHTS_DIR, "moondream2")

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
VLM_REPO = "vikhyatk/moondream2"

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

    # 2. Download Moondream2 (VLM)
    if not os.path.exists(VLM_PATH):
        print(f"Downloading Moondream2 (VLM) snapshot to {VLM_PATH}...")
        snapshot_download(repo_id=VLM_REPO, local_dir=VLM_PATH, revision="2024-03-06")
    else:
        print(f"VLM already exists at {VLM_PATH}")

    print("\n[SUCCESS] Intelligence models downloaded.")

if __name__ == "__main__":
    main()
