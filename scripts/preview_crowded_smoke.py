import cv2
import numpy as np
from pathlib import Path

# Config
BASE_DIR   = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight")
IMG_DIR    = BASE_DIR / "Dataset" / "Indian Dataset" / "Crowded_Condition"
LBL_DIR    = BASE_DIR / "Dataset" / "Final_Annotated_Dataset" / "labels" / "train"
OUT_FILE   = BASE_DIR / "Dataset" / "crowded_smoke_preview.jpg"

# 3-class schema
COLORS = {
    0: (0, 255, 255),  # helmet (yellow)
    1: (0, 165, 255),  # vest (orange)
    2: (255, 0, 255),  # worker (magenta)
}
NAMES = {0: "helmet", 1: "vest", 2: "worker"}

def draw_yolo(img, lbl_path):
    h, w, _ = img.shape
    if not lbl_path.exists():
        return img
        
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        
        cls = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        
        # Denormalize
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        color = COLORS.get(cls, (255,255,255))
        label = NAMES.get(cls, f"cls_{cls}")
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def main():
    # Hardcode stems from the smoke test run
    stems = ["1772974215897", "1772974215985", "1772974216097", "1772974216253"]
    
    # Store in artifacts for easy embedding
    artifact_dir = Path(r"C:\Users\brigh\.gemini\antigravity\brain\9ad522c6-e50c-4b25-bbf4-ca1c61a23d9b")
    
    for i, stem in enumerate(stems):
        lbl_p = LBL_DIR / (stem + ".txt")
        img_p = IMG_DIR / (stem + ".jpg")
        
        if not lbl_p.exists() or not img_p.exists():
            print(f"Skipping {stem} - files missing")
            continue
        
        img = cv2.imread(str(img_p))
        if img is None: continue
        
        img = draw_yolo(img, lbl_p)
        out_path = artifact_dir / f"crowded_smoke_{i}.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
