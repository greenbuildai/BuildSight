import os
import random
import numpy as np
import cv2
from PIL import Image

SRC_NORMAL = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition"
DST_DUSTY = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"

def generate_dusty(img_path, output_path):
    # Load with OpenCV for haze effect
    img = cv2.imread(img_path)
    if img is None: return False
    
    # 1. Overlay a brown/gray haze
    haze_color = np.array([180, 200, 220], dtype=np.uint8) # Dusty brown in BGR
    # If daytime, maybe lighter sand color
    if random.random() > 0.5:
        haze_color = np.array([210, 230, 240], dtype=np.uint8)
        
    overlay = np.full(img.shape, haze_color, dtype=np.uint8)
    alpha = random.uniform(0.30, 0.55) # Haze intensity slightly increased for distinct dusty look
    img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    
    # 2. Subtle Blur (dust particles scattering light)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 3. Save
    cv2.imwrite(output_path, img)
    return True

def main():
    os.makedirs(DST_DUSTY, exist_ok=True)
    
    files = [f for f in os.listdir(SRC_NORMAL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    
    # Check how many images we currently have in dusty directory
    existing_dusty = len([f for f in os.listdir(DST_DUSTY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    target_count = 500
    
    print(f"Current Dusty Images: {existing_dusty}")
    print(f"Targeting: 500 New Augmented Dusty Images")
    
    generated_count = 0
    
    start_offset = 1500  # Avoid writing over "augmented_dusty_0" using a large offset
    for i, file in enumerate(files):
        if generated_count >= target_count:
            break
            
        src = os.path.join(SRC_NORMAL, file)
        
        # Save explicitly as SYNTHETIC representation so it maps back without issues
        dst = os.path.join(DST_DUSTY, f"synthetic_opencv_dusty_{start_offset+i}_{file}")
        
        success = generate_dusty(src, dst)
        if success:
            generated_count += 1
            if generated_count % 50 == 0: 
                print(f"  Dusty Augmented: {generated_count}/{target_count}")

    print(f"\n=== Augmentation Complete. Generated {generated_count} new dusty condition images ===")

if __name__ == "__main__":
    main()
