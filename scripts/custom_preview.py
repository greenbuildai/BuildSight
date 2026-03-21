import os
import cv2
import glob
import numpy as np

base_dir = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset"
img_dir = os.path.join(base_dir, "images", "train")
lbl_dir = os.path.join(base_dir, "labels", "train")
imgs = list(glob.glob(os.path.join(img_dir, "*.jpg")))
np.random.shuffle(imgs)

# Colors per class (BGR for cv2) — 3-class sequential schema
colors = {0: (0, 200, 255), 1: (0, 220, 80), 2: (200, 40, 220)}

out_imgs = []
for p in imgs:
    img = cv2.imread(p)
    if img is None: continue
    h, w = img.shape[:2]
    
    lbl_file = os.path.join(lbl_dir, os.path.basename(p).replace(".jpg", ".txt"))
    if os.path.exists(lbl_file):
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    color = colors.get(cls_id, (255, 255, 255))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    label_text = "helmet" if cls_id == 0 else "vest" if cls_id == 1 else "worker"
                    cv2.putText(img, label_text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
    # Resize to box for grid
    img = cv2.resize(img, (600, 600))
    out_imgs.append(img)
    if len(out_imgs) == 4: break

if len(out_imgs) == 4:
    top = np.hstack((out_imgs[0], out_imgs[1]))
    bot = np.hstack((out_imgs[2], out_imgs[3]))
    grid = np.vstack((top, bot))
    cv2.imwrite(r"C:\Users\brigh\.gemini\antigravity\brain\7cfec27d-6f66-4e22-bbbe-acd88b9bd8ae\live_yolo_preview_grid.png", grid)
    print("Preview grid saved successfully to artifacts!")
else:
    print(f"Not enough images found to render grid: found {len(out_imgs)}")
