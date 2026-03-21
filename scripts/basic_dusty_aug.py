import os
import cv2
import random
import numpy as np

src_folder = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"

def apply_standard_augmentation(img):
    # Apply a random mix of standard digital augmentations
    
    # 1. Random Horizontal Flip (50% chance)
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # 2. Random Brightness / Contrast tuning (simulate slight lighting change)
    alpha = random.uniform(0.8, 1.2) # Contrast control (1.0-1.3)
    beta = random.randint(-15, 15)   # Brightness control (0-100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 3. Random Gaussian Blur (Simulating slight camera blur or heavier dust)
    if random.random() > 0.5:
        kernel_size = random.choice([(3,3), (5,5)])
        img = cv2.GaussianBlur(img, kernel_size, 0)

    # 4. Slight rotation
    if random.random() > 0.5:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return img

if __name__ == "__main__":
    # Get all original dusty images (excluding synthetics)
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
             and not f.startswith(('synthetic_', 'advanced_', 'digi_aug_'))]
    
    # Select 10 random original images
    sample_files = random.sample(files, min(10, len(files)))
    
    generated = 0
    for file in sample_files:
        path = os.path.join(src_folder, file)
        img = cv2.imread(path)
        
        if img is not None:
            aug_img = apply_standard_augmentation(img)
            dst_path = os.path.join(src_folder, f"digi_aug_{generated}_{file}")
            cv2.imwrite(dst_path, aug_img)
            print(f"Generated -> {dst_path}")
            generated += 1

    print(f"Successfully applied digital augmentation to {generated} original dusty photos.")
