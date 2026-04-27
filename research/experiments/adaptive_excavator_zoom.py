"""Zoom into the excavator-heavy rows of the comparison."""
import cv2, numpy as np
from pathlib import Path

FIXED_DIR    = Path("e:/Company/Green Build AI/Prototypes/BuildSight/val_annotated_fixed/S1_normal/YOLOv11")
ADAPTIVE_DIR = Path("e:/Company/Green Build AI/Prototypes/BuildSight/val_annotated_adaptive/S1_normal/YOLOv11")
OUT = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/adaptive_excavator_zoom.jpg")

# Score files by excavator content in FIXED (more orange boxes on top = more machinery FP)
def score(img_bgr):
    h = img_bgr.shape[0]
    upper = img_bgr[:h//2]
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (15,100,100),(38,255,255)).sum()

files = sorted(FIXED_DIR.glob("*.jpg"))
scored = sorted([(score(cv2.imread(str(f))), f.name) for f in files if cv2.imread(str(f)) is not None], reverse=True)
top = [n for _, n in scored[:5]]

rows = []
for fname in top:
    f = cv2.imread(str(FIXED_DIR / fname))
    a = cv2.imread(str(ADAPTIVE_DIR / fname))
    if f is None or a is None: continue
    f = cv2.resize(f, (900, 675))
    a = cv2.resize(a, (900, 675))
    div = np.full((675, 8, 3), 255, np.uint8)
    # Labels
    cv2.rectangle(f, (0,0),(900,26),(0,50,160),-1)
    cv2.putText(f,"FIXED (FIX-A+B+C)",(8,19),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),1)
    cv2.rectangle(a,(0,0),(900,26),(0,120,0),-1)
    cv2.putText(a,"ADAPTIVE (7-rule)",(8,19),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),1)
    rows.append(np.hstack([f, div, a]))

grid = np.vstack(rows)
cv2.imwrite(str(OUT), grid, [cv2.IMWRITE_JPEG_QUALITY,90])
print(f"Saved {OUT}")
