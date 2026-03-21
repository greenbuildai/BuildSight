import os
import sys
import subprocess

try:
    import imagehash
except ImportError:
    print("Installing imagehash...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imagehash"])
    import imagehash

from PIL import Image

BASE_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp')

# The threshold for how similar two images need to be to be considered "duplicates"
# The hashing generates a hex string. A difference (Hamming distance) of <= 4 means the images are basically identical.
SIMILARITY_THRESHOLD = 4 

def visual_audit():
    print(f"Starting Visual AI Audit in: {BASE_DIR}")
    
    Stats = {
        'total_scanned': 0,
        'removed_visual_duplicates': 0,
    }
    
    # Store hashes to compare: { condition_folder : [ (filepath, phash_value) ] }
    # We check for duplicates within the entire dataset to prevent the same image appearing in multiple conditions
    global_hashes = []

    for root, dirs, files in os.walk(BASE_DIR):
        if "Extracted_Frames" in root or "labels" in root:
            continue
            
        for file in files:
            filepath = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            
            if ext not in SUPPORTED_EXTENSIONS:
                continue
                
            Stats['total_scanned'] += 1
            
            try:
                # Perceptual hash uses Computer Vision (frequency domain analysis via DCT) 
                # to generate a fingerprint based on visual structure, ignoring small color/lighting changes or compression
                with Image.open(filepath) as img:
                    # phash (Perceptual Hash) is robust to resizing, compression, and slight framing differences
                    img_hash = imagehash.phash(img)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue
                
            is_duplicate = False
            for prev_filepath, prev_hash in global_hashes:
                # Calculate Hamming distance between hashes
                if img_hash - prev_hash <= SIMILARITY_THRESHOLD:
                    print(f"Visual Duplicate Found: \n  Keep: {os.path.basename(prev_filepath)}\n  Delete: {file}\n")
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                try:
                    os.remove(filepath)
                    Stats['removed_visual_duplicates'] += 1
                except Exception as e:
                    print(f"Could not delete {filepath}: {e}")
            else:
                global_hashes.append((filepath, img_hash))
                
    print("\n=== Visual AI Audit Complete ===")
    print(f"Total Images Scanned: {Stats['total_scanned']}")
    print(f"Visually Identical Duplicates Removed: {Stats['removed_visual_duplicates']}")
    print(f"Final Clean Dataset Count: {Stats['total_scanned'] - Stats['removed_visual_duplicates']}")

if __name__ == "__main__":
    visual_audit()
