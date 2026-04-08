"""
compare_fixed_vs_adaptive.py
Picks the worst excavator/crane scenes from S1_normal YOLOv11
and generates a FIXED vs ADAPTIVE side-by-side comparison grid.
"""
import cv2
import numpy as np
from pathlib import Path

FIXED_DIR    = Path("e:/Company/Green Build AI/Prototypes/BuildSight/val_annotated_fixed/S1_normal/YOLOv11")
ADAPTIVE_DIR = Path("e:/Company/Green Build AI/Prototypes/BuildSight/val_annotated_adaptive/S1_normal/YOLOv11")
OUT          = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/compare_fixed_vs_adaptive.jpg")

all_files = sorted(FIXED_DIR.glob("*.jpg"))

# Score each image by how many large orange/yellow boxes it has (machinery FP indicator)
# Use the FIXED image - count pixels in the orange/blue box color range in top half
def machinery_score(img_bgr):
    h = img_bgr.shape[0]
    upper = img_bgr[:h//2]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 100, 100), (38, 255, 255))
    return yellow.sum()

scored = []
for f in all_files:
    img = cv2.imread(str(f))
    if img is None: continue
    scored.append((machinery_score(img), f.name))

scored.sort(reverse=True)
top_names = [s[1] for s in scored[:8]]

rows = []
for fname in top_names:
    fixed_path    = FIXED_DIR / fname
    adaptive_path = ADAPTIVE_DIR / fname
    if not fixed_path.exists() or not adaptive_path.exists():
        continue

    fixed    = cv2.resize(cv2.imread(str(fixed_path)),    (640, 480))
    adaptive = cv2.resize(cv2.imread(str(adaptive_path)), (640, 480))

    div = np.full((480, 6, 3), 200, dtype=np.uint8)
    row = np.hstack([fixed, div, adaptive])
    rows.append(row)

if not rows:
    print("No images found")
    exit()

# Header
grid = np.vstack(rows)
header = np.zeros((40, grid.shape[1], 3), np.uint8)
cv2.putText(header, "FIXED (FIX-A+B+C only)", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 180, 255), 2)
cv2.putText(header, "ADAPTIVE (7-rule system)", (666, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
grid = np.vstack([header, grid])

cv2.imwrite(str(OUT), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(f"Saved -> {OUT}  ({len(rows)} rows)")
