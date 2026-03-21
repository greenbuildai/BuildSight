import os
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageStat
from pathlib import Path

BASE_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def calculate_hash(filepath):
    """Calculates MD5 hash of a file for exact duplicate detection."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def audit_dataset():
    print(f"Starting Dataset Audit in: {BASE_DIR}")
    
    Stats = {
        'total_scanned': 0,
        'converted_to_jpg': 0,
        'removed_corrupted': 0,
        'removed_duplicates': 0,
        'removed_low_res': 0,
        'removed_blurry': 0
    }
    
    seen_hashes = set()

    for root, dirs, files in os.walk(BASE_DIR):
        # Skip the Extracted_Frames folder for this audit, or output folders
        if "Extracted_Frames" in root or "labels" in root:
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext not in SUPPORTED_EXTENSIONS:
                continue
                
            Stats['total_scanned'] += 1
            
            # 1. Duplicate Check
            file_hash = calculate_hash(filepath)
            if file_hash is None:
                print(f"[{root}] Error reading: {file}")
                continue
                
            if file_hash in seen_hashes:
                print(f"Removing DUPLICATE: {file}")
                os.remove(filepath)
                Stats['removed_duplicates'] += 1
                continue
                
            seen_hashes.add(file_hash)
            
            # 2. Corrupted / Low Res / Format Check
            try:
                with Image.open(filepath) as img:
                    img.verify() # Verify integrity
                
                # Re-open for actual processing (verify closes it)
                with Image.open(filepath) as img:
                    width, height = img.size
                    
                    if width < 120 or height < 120:
                        print(f"Removing LOW RES ({width}x{height}): {file}")
                        os.remove(filepath)
                        Stats['removed_low_res'] += 1
                        continue
                        
                    # 3. Standardize to RGB .jpg
                    converted = False
                    if img.mode != 'RGB' or ext != '.jpg':
                        # Convert and save as JPG
                        rgb_im = img.convert('RGB')
                        new_filename = os.path.splitext(filepath)[0] + ".jpg"
                        
                        rgb_im.save(new_filename, "JPEG", quality=95)
                        converted = True
                        
                        # Remove original if it was a different extension
                        if filepath != new_filename:
                            os.remove(filepath)
                            file = os.path.basename(new_filename) # Update reference
                            filepath = new_filename
                            
                        Stats['converted_to_jpg'] += 1
                        
            except Exception as e:
                print(f"Removing CORRUPTED file {file}: {e}")
                os.remove(filepath)
                Stats['removed_corrupted'] += 1
                continue
                
    print("\n=== Audit Complete ===")
    for k, v in Stats.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    audit_dataset()
