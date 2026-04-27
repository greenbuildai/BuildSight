import os
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter

SRC_NORMAL = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition"
DST_LOW_LIGHT = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Low_Light_Condition"
DST_DUSTY = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"
DST_CROWDED = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Crowded_Condition"

def generate_low_light(img_path, output_path):
    # Load image
    img = Image.open(img_path)
    
    # 1. Reduce Brightness significantly (30-50%)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.15, 0.35))
    
    # 2. Adjust Contrast (compensate for dimness)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 3. Add Color Jitter (blues/oranges for artificial light)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.5, 0.8))
    
    # 4. Add subtle noise (simulating high ISO)
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    # 5. Save
    Image.fromarray(arr).save(output_path, "JPEG", quality=85)

def generate_dusty(img_path, output_path):
    # Load with OpenCV for haze effect
    img = cv2.imread(img_path)
    if img is None: return
    
    # 1. Overlay a brown/gray haze
    haze_color = np.array([180, 200, 220], dtype=np.uint8) # Dusty brown in BGR
    # If daytime, maybe lighter sand color
    if random.random() > 0.5:
        haze_color = np.array([210, 230, 240], dtype=np.uint8)
        
    overlay = np.full(img.shape, haze_color, dtype=np.uint8)
    alpha = random.uniform(0.25, 0.45) # Haze intensity
    img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    
    # 2. Subtle Blur (dust particles scattering light)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Save
    cv2.imwrite(output_path, img)

def generate_crowded_view(img_path, output_path):
    # Simulate a "crowded" feel by zooming in on worker groups or cropping
    img = Image.open(img_path)
    w, h = img.size
    
    # Random Crop (zoom in by 20-40%) to make people appear larger/more packed
    zoom_factor = random.uniform(0.6, 0.8)
    new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    
    img = img.crop((left, top, left + new_w, top + new_h))
    img.save(output_path, "JPEG", quality=90)

def main():
    os.makedirs(DST_LOW_LIGHT, exist_ok=True)
    os.makedirs(DST_DUSTY, exist_ok=True)
    os.makedirs(DST_CROWDED, exist_ok=True)
    
    files = [f for f in os.listdir(SRC_NORMAL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    
    print(f"Starting Digital Augmentation on {len(files)} source images...")
    
    # 1. Generate 350 Low Light
    for i in range(min(350, len(files))):
        src = os.path.join(SRC_NORMAL, files[i])
        dst = os.path.join(DST_LOW_LIGHT, f"augmented_lowlight_{i}.jpg")
        generate_low_light(src, dst)
        if i % 50 == 0: print(f"  Low Light: {i}/350")
        
    # 2. Generate 100 Dusty
    for i in range(min(100, len(files))):
        src = os.path.join(SRC_NORMAL, files[i])
        dst = os.path.join(DST_DUSTY, f"augmented_dusty_{i}.jpg")
        generate_dusty(src, dst)
        if i % 25 == 0: print(f"  Dusty: {i}/100")

    # 3. Generate 150 Crowded
    for i in range(min(150, len(files))):
        src = os.path.join(SRC_NORMAL, files[i])
        dst = os.path.join(DST_CROWDED, f"augmented_crowded_{i}.jpg")
        generate_crowded_view(src, dst)
        if i % 40 == 0: print(f"  Crowded: {i}/150")

    print("\n=== Augmentation Complete ===")

if __name__ == "__main__":
    main()
