import os
import cv2
import glob
import numpy as np
import random

img_dir = r"Dataset\Final_Annotated_Dataset\images\train"
lbl_dir = r"Dataset\Final_Annotated_Dataset\labels\train"
output_path = r"normal_annotation_preview.png"

colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255), 3: (255, 0, 0), 6: (255, 0, 255)}
names = {0: "helmet", 1: "safety_vest", 2: "reserved", 3: "safety_boots", 6: "worker"}

# Find images that belong to Normal_Site_Condition
all_images = glob.glob(os.path.join(img_dir, "*Normal_Site_Condition*.jpg"))
if not all_images:
    # fallback to any image if the naming schema doesn't contain the condition name
    all_images = glob.glob(os.path.join(img_dir, "*.jpg"))

# Pick 6 random images
random.shuffle(all_images)
images = all_images[:6]

if not images:
    print("No images found to preview!")
    exit(1)

sample = cv2.imread(images[0])
target_h, target_w = 640, 853

rows = 2
cols = 3
canvas_w = target_w * cols
canvas_h = target_h * rows
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

row, col = 0, 0
for img_path in images:
    img = cv2.imread(img_path)
    if img is None: continue
    
    h_img, w_img = img.shape[:2]
    
    lbl_path = os.path.join(lbl_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    
                    x1 = int((x - w/2) * w_img)
                    y1 = int((y - h/2) * h_img)
                    x2 = int((x + w/2) * w_img)
                    y2 = int((y + h/2) * h_img)
                    
                    color = colors.get(c, (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    label = names.get(c, str(c))
                    
                    cv2.rectangle(img, (x1, max(0, y1 - 30)), (x1 + len(label)*15, y1), color, -1)
                    cv2.putText(img, label, (x1+5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    img = cv2.resize(img, (target_w, target_h))
    canvas[row*target_h:(row+1)*target_h, col*target_w:(col+1)*target_w] = img
    col += 1
    if col >= cols:
        col = 0
        row += 1

cv2.imwrite(output_path, canvas)
print(f"Saved preview of normal annotations to {output_path}")
