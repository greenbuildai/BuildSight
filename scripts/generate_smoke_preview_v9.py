import os
import cv2
import glob
import numpy as np
import random

img_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\images\train"
lbl_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\labels\train"
output_path = r"e:\Company\Green Build AI\Prototypes\BuildSight\v9_normal_preview.png"

colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 0, 255)}  # magenta worker, yellow helmet, green vest
names = {0: "helmet", 1: "vest", 2: "worker"}

# Force Normal_Site_Condition only
images = list(glob.glob(os.path.join(img_dir, "*Normal_Site_Condition*.jpg")))
random.shuffle(images)
images = images[:6]

if not images:
    print("No Normal_Site_Condition images found in", img_dir)
    exit(1)

target_h, target_w = 640, 853
header_h = 40  # Room for the header text

rows = 2
cols = 3
canvas_w = target_w * cols
canvas_h = (target_h * rows) + header_h
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

# Draw the exact V9 header
header_text = "BuildSight | Smoke Test V9 | Normal_Site_Condition | helmet-position + vest-size + no-conf fixes"
cv2.putText(canvas, header_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

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
                    if c == 1: n_v += 1
                    if c == 2: n_w += 1
                    
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

    # Draw individual image top bar stats like in the screenshot
    cv2.putText(img, f"[TRAIN] workers={n_w} helmets={n_h} vests={n_v}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

    # Resize to fit grid
    img = cv2.resize(img, (target_w, target_h))
    
    start_y = header_h + (row * target_h)
    start_x = col * target_w
    canvas[start_y:start_y+target_h, start_x:start_x+target_w] = img
    
    col += 1
    if col >= cols:
        col = 0
        row += 1

cv2.imwrite(output_path, canvas)
print(f"Saved preview to {output_path}")
