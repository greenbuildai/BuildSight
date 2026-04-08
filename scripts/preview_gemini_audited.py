import cv2
import sys
import glob
import os
from pathlib import Path

IMG_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\images\train")
LBL_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\labels\train")
OUT_DIR = Path(r"c:\Users\brigh\.gemini\antigravity\brain\9ad522c6-e50c-4b25-bbf4-ca1c61a23d9b")

CLASS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 255), 2: (255, 0, 0)} # BGR: helmet=red, vest=yellow, worker=blue

target_images = ["1772974217056.jpg", "1772974480488.jpg"]

for img_name in target_images:
    img_path = IMG_DIR / img_name
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    
    if not img_path.exists() or not lbl_path.exists():
        print(f"Skipping {img_name}, files not found: {img_path} / {lbl_path}")
        continue
        
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    with open(lbl_path, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # yolo to rect
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            
            color = COLORS.get(cls_id, (255, 255, 255))
            label = CLASS_NAMES.get(cls_id, f"ID_{cls_id}")
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
            cv2.putText(img, label, (x1, max(15, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    out_file = OUT_DIR / f"annotated_preview_{img_name}"
    cv2.imwrite(str(out_file), img)
    print(f"Saved: {out_file}")
