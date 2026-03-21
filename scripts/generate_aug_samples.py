import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance

# Paths
SRC_NORMAL = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition"
OUTPUT_DIR = r"C:\Users\brigh\.gemini\antigravity\brain\d28238d3-3806-46aa-aa80-57372b7de723"

def apply_low_light(img):
    # Brightness reduction
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.25)
    # Add noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def apply_dusty(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    haze_color = np.array([180, 200, 220], dtype=np.uint8) 
    overlay = np.full(img.shape, haze_color, dtype=np.uint8)
    alpha = 0.35
    img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def apply_crowded(img):
    w, h = img.size
    zoom = 0.65
    new_w, new_h = int(w * zoom), int(h * zoom)
    left, top = (w - new_w)//2, (h - new_h)//2 # Center crop for sample
    return img.crop((left, top, left + new_w, top + new_h))

def main():
    files = [f for f in os.listdir(SRC_NORMAL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print("No files found!")
        return
    
    sample_file = random.choice(files)
    src_path = os.path.join(SRC_NORMAL, sample_file)
    print(f"Using sample: {sample_file}")

    # 1. Original
    img_pil = Image.open(src_path)
    img_pil.save(os.path.join(OUTPUT_DIR, "sample_original.jpg"))

    # 2. Low Light
    low_light = apply_low_light(img_pil)
    low_light.save(os.path.join(OUTPUT_DIR, "sample_lowlight.jpg"))

    # 3. Dusty
    dusty = apply_dusty(src_path)
    if dusty is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, "sample_dusty.jpg"), dusty)

    # 4. Crowded
    crowded = apply_crowded(img_pil)
    crowded.save(os.path.join(OUTPUT_DIR, "sample_crowded.jpg"))

    print("Samples generated in artifact directory.")

if __name__ == "__main__":
    main()
