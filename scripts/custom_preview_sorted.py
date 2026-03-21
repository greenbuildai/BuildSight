import os
import cv2
import glob
import numpy as np

base_dir = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset"
img_dir = os.path.join(base_dir, "images", "train")
lbl_dir = os.path.join(base_dir, "labels", "train")

# Find the newest YOLO label files
txt_files = list(glob.glob(os.path.join(lbl_dir, "*.txt")))
txt_files.sort(key=os.path.getmtime, reverse=True)
newest_labels = txt_files[:100]  # Check top 100 newest
np.random.shuffle(newest_labels)

colors = {0: (0, 200, 255), 1: (0, 220, 80), 2: (200, 40, 220)}

out_imgs = []
for lbl_file in newest_labels:
    img_path = os.path.join(img_dir, os.path.basename(lbl_file).replace(".txt", ".jpg"))
    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w = img.shape[:2]
    
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
                
    img = cv2.resize(img, (600, 600))
    out_imgs.append(img)
    if len(out_imgs) == 9: break

if len(out_imgs) == 9:
    r1 = np.hstack((out_imgs[0], out_imgs[1], out_imgs[2]))
    r2 = np.hstack((out_imgs[3], out_imgs[4], out_imgs[5]))
    r3 = np.hstack((out_imgs[6], out_imgs[7], out_imgs[8]))
    grid = np.vstack((r1, r2, r3))
    cv2.imwrite(r"C:\Users\brigh\.gemini\antigravity\brain\7cfec27d-6f66-4e22-bbbe-acd88b9bd8ae\live_yolo_preview_newest.png", grid)
    print("Preview grid saved successfully to artifacts!")
else:
    print(f"Not enough recent images found to render grid: found {len(out_imgs)}")
