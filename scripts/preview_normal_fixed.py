import os
import cv2
import glob
import numpy as np
import random
from pathlib import Path

# Paths
img_dir = r"Dataset\Final_Annotated_Dataset\images\train"
lbl_dir = r"Dataset\Final_Annotated_Dataset\labels\train"
full_lbl_root = r"Dataset\Final_Annotated_Dataset\labels"
output_path = r"fixed_normal_annotation_preview.jpg" # Changed to JPG for 5MB limit

# Progress calculation (like in the screenshot)
TOTAL_IMAGES = 1373
current_count = 0
if os.path.exists(full_lbl_root):
    for root, dirs, files in os.walk(full_lbl_root):
        current_count += len([f for f in files if f.endswith(".txt")])

progress_pct = (current_count / TOTAL_IMAGES) * 100

# 3-class schema
colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 0, 255)} # helmet: green, vest: orange, worker: magenta
names = {0: "helmet", 1: "safety_vest", 2: "worker"}

all_images = glob.glob(os.path.join(img_dir, "*Normal_Site_Condition*.jpg"))
if not all_images:
    all_images = glob.glob(os.path.join(img_dir, "*.jpg"))

random.shuffle(all_images)
images = all_images[:6]

if not images:
    print("No images found to preview!")
    exit(1)

target_h, target_w = 640, 853
header_h = 60

rows, cols = 2, 3
canvas_w = target_w * cols
canvas_h = (target_h * rows) + header_h
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Draw the requested header
header_text = f"BuildSight | NormaL Site_Condition | {current_count}/{TOTAL_IMAGES} ({progress_pct:.1f}%) | Set 2 preview"
cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (40, 40, 40), -1)
cv2.putText(canvas, header_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

row, col = 0, 0
for img_path in images:
    img = cv2.imread(img_path)
    if img is None: continue
    
    h_img, w_img = img.shape[:2]
    
    n_w, n_h, n_v = 0, 0, 0
    lbl_path = os.path.join(lbl_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    if c == 0: n_h += 1
                    elif c == 1: n_v += 1
                    elif c == 2: n_w += 1
                    
                    x, y, w, h = map(float, parts[1:5])
                    x1 = int((x - w/2) * w_img)
                    y1 = int((y - h/2) * h_img)
                    x2 = int((x + w/2) * w_img)
                    y2 = int((y + h/2) * h_img)
                    
                    color = colors.get(c, (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    label = names.get(c, str(c))
                    
                    cv2.rectangle(img, (x1, max(0, y1 - 35)), (x1 + len(label)*20, y1), color, -1)
                    cv2.putText(img, label, (x1+5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

    # Sub-header per image
    sub_text = f"[TRAIN] workers={n_w} helmets={n_h} vests={n_v} | {os.path.basename(img_path)}"
    cv2.putText(img, sub_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    img = cv2.resize(img, (target_w, target_h))
    y_off = header_h + row * target_h
    x_off = col * target_w
    canvas[y_off:y_off+target_h, x_off:x_off+target_w] = img
    
    col += 1
    if col >= cols:
        col = 0
        row += 1

# Save as JPEG with quality control to ensure < 5MB (usually ~1MB at this size)
cv2.imwrite(output_path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
print(f"Saved optimized progress preview to {output_path}")

