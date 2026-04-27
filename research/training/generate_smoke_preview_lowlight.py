import os
import cv2
import numpy as np
import random
from pathlib import Path

# Paths
IMG_DIR = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\images\train"
LBL_DIR = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\labels\train"
SRC_LOWLIGHT_DIR = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Low_Light_Condition"
OUTPUT_PATH = r"e:\Company\Green Build AI\Prototypes\BuildSight\v18_lowlight_preview.png"

colors = {0: (0, 255, 0), 1: (0, 165, 255), 2: (0, 0, 255)}  # red worker, orange vest, green helmet
names = {0: "helmet", 1: "vest", 2: "worker"}

def generate_preview():
    # Identify images that belong to Low_Light_Condition
    lowlight_stems = [f.stem for f in Path(SRC_LOWLIGHT_DIR).glob("*.jpg")]
    
    # Filter available
    available = [s for s in lowlight_stems if (Path(IMG_DIR)/(s+".jpg")).exists()]
    if not available:
        print("No annotated images found yet in train! Checking val/test...")
        img_val = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\images\val"
        available = [s for s in lowlight_stems if (Path(img_val)/(s+".jpg")).exists()]
        if not available:
            print("No annotated low light images found!")
            return 
        else:
            global IMG_DIR, LBL_DIR
            IMG_DIR = img_val
            LBL_DIR = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset\labels\val"

    # Only show up to 6 images
    random.seed(42)  # Consistency for the smoke test
    random.shuffle(available)
    display_stems = available[:6]

    target_h, target_w = 640, 853
    header_h = 45 
    rows, cols = 2, 3
    canvas_w = target_w * cols
    canvas_h = (target_h * rows) + header_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Header
    header_text = f"BuildSight | Low_Light v18 Fix Test | {len(available)} Found"
    cv2.putText(canvas, header_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, stem in enumerate(display_stems):
        img_path = str(Path(IMG_DIR) / (stem + ".jpg"))
        lbl_path = str(Path(LBL_DIR) / (stem + ".txt"))
        
        # Check other splits if not found in current directory
        if not os.path.exists(img_path):
             for s in ["test", "val", "train"]:
                 img_path_s = img_path.replace("val", s).replace("train", s)
                 lbl_path_s = lbl_path.replace("val", s).replace("train", s)
                 if os.path.exists(img_path_s): 
                     img_path = img_path_s
                     lbl_path = lbl_path_s
                     break

        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        counts = {0:0, 1:0, 2:0}
        has_label = os.path.exists(lbl_path)
        if has_label:
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    cid = int(float(parts[0]))
                    counts[cid] = counts.get(cid, 0) + 1
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw/2) * w_img)
                    y1 = int((cy - bh/2) * h_img)
                    x2 = int((cx + bw/2) * w_img)
                    y2 = int((cy + bh/2) * h_img)
                    
                    color = colors.get(cid, (255, 255, 255))
                    name = names.get(cid, str(cid))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    
                    # put text indicating class above box
                    cv2.putText(img, name, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Overlay Stats
        status_c = (0, 255, 0) if has_label else (0, 0, 255)
        stat_str = f"{stem}"
        if not has_label: stat_str += " (WAITING)"
        
        cv2.rectangle(img, (0,0), (target_w, 40), (0,0,0), -1)
        cv2.putText(img, f"{stat_str} | W:{counts[2]} H:{counts[0]} V:{counts[1]}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_c, 2)

        img_rs = cv2.resize(img, (target_w, target_h))
        row, col = divmod(i, cols)
        start_y = header_h + row * target_h
        start_x = col * target_w
        canvas[start_y:start_y+target_h, start_x:start_x+target_w] = img_rs

    cv2.imwrite(OUTPUT_PATH, canvas)
    print(f"Saved preview: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_preview()
