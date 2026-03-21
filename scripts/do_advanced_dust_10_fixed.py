import os
import cv2
import random
import numpy as np

def apply_dusty(img_path):
    img = cv2.imread(img_path)
    if img is None: return None
    
    # Use the hazy beige color from the previous sample
    haze_color = np.array([180, 200, 220], dtype=np.uint8) 
    overlay = np.full(img.shape, haze_color, dtype=np.uint8)
    
    # 35% opacity wash
    alpha = 0.35
    img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    
    # Add blur to simulate hazy air
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

if __name__ == "__main__":
    src_folder = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition"
    dst_folder = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"

    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    generated = 0
    for f in files:
        src_path = os.path.join(src_folder, f)
        dst_path = os.path.join(dst_folder, f"advanced_dust_aug_V3_{generated}_{f}")
        
        result_img = apply_dusty(src_path)
        if result_img is not None:
            cv2.imwrite(dst_path, result_img)
            print(f"Generated {dst_path}")
            generated += 1
            
        if generated >= 10:
            break

    print(f"Successfully generated {generated} advanced dust augmented images.")
