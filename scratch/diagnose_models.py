import os
import sys
import torch
from pathlib import Path

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "dashboard", "backend")))

def check_sam():
    print("--- SAM Check ---")
    try:
        import segment_anything
        print("segment_anything package: OK")
    except ImportError:
        print("segment_anything package: MISSING")
        return

    ckpt_path = "dashboard/backend/weights/sam_vit_b.pth"
    if os.path.exists(ckpt_path):
        print(f"SAM checkpoint found: {ckpt_path} ({os.path.getsize(ckpt_path)} bytes)")
        try:
            from segment_anything import sam_model_registry
            sam = sam_model_registry["vit_b"](checkpoint=ckpt_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device=device)
            print(f"SAM loaded successfully on {device}")
        except Exception as e:
            print(f"SAM loading failed: {e}")
    else:
        print(f"SAM checkpoint NOT FOUND: {ckpt_path}")

def check_vlm():
    print("\n--- VLM Check ---")
    try:
        import transformers
        print("transformers package: OK")
    except ImportError:
        print("transformers package: MISSING")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "vikhyatk/moondream2"
    local_path = "dashboard/backend/weights/moondream2"
    
    # Check if local files exist
    if os.path.exists(local_path):
        print(f"VLM local path found: {local_path}")
        try:
            # Try loading from local path
            model = AutoModelForCausalLM.from_pretrained(local_path, trust_remote_code=True)
            print("VLM loaded successfully from local weights")
        except Exception as e:
            print(f"VLM loading from local failed: {e}")
            print("Checking remote/cache fallback...")
            try:
                # Try loading from HF Hub (will use cache if available)
                model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision="2024-08-26")
                print("VLM loaded successfully from HF Hub/Cache")
            except Exception as e2:
                print(f"VLM loading from HF Hub failed: {e2}")
    else:
        print(f"VLM local path NOT FOUND: {local_path}")

if __name__ == "__main__":
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    check_sam()
    check_vlm()
