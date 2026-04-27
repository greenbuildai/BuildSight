"""Compare specific images that had excavator FPs in the original fixed output."""
import cv2, numpy as np
from pathlib import Path

FIXED_DIR    = Path("e:/Company/Green Build AI/Prototypes/BuildSight/val_annotated_fixed/S1_normal/YOLOv11")
ADAPTIVE_DIR = Path("e:/Company/Green Build AI/Prototypes/BuildSight/val_annotated_adaptive/S1_normal/YOLOv11")
OUT = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/adaptive_specific_compare.jpg")

# Find images that have sky+excavator scenes: outdoor with yellow machinery
# Filter by: outdoor (sky in top 20%) + yellow blob present
def is_outdoor_excavator(img_bgr):
    h, w = img_bgr.shape[:2]
    upper = img_bgr[:h//5]
    hsv_upper = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    sky = cv2.inRange(hsv_upper, (90,30,120),(135,255,255)).sum()
    bright = cv2.inRange(upper, (180,180,180),(255,255,255)).sum()
    sky_frac = (sky + bright) / (255 * upper.shape[0] * upper.shape[1])

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15,120,100),(38,255,255)).sum()
    yel_frac = yellow / (255 * h * w)

    return sky_frac > 0.08 and yel_frac > 0.04

files = sorted(FIXED_DIR.glob("*.jpg"))
candidates = []
for f in files:
    img = cv2.imread(str(f))
    if img is not None and is_outdoor_excavator(img):
        candidates.append(f.name)

print(f"Found {len(candidates)} outdoor excavator scenes")
selected = candidates[:6]

rows = []
for fname in selected:
    f = cv2.imread(str(FIXED_DIR / fname))
    a = cv2.imread(str(ADAPTIVE_DIR / fname))
    if f is None or a is None: continue
    f = cv2.resize(f, (860, 645))
    a = cv2.resize(a, (860, 645))
    div = np.full((645, 8, 3), 255, np.uint8)
    cv2.rectangle(f,(0,0),(860,24),(0,50,180),-1)
    cv2.putText(f,"FIXED",(8,17),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    cv2.rectangle(a,(0,0),(860,24),(0,140,0),-1)
    cv2.putText(a,"ADAPTIVE",(8,17),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    rows.append(np.hstack([f, div, a]))

if rows:
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT), grid, [cv2.IMWRITE_JPEG_QUALITY,90])
    print(f"Saved -> {OUT}")
else:
    print("No matching images found")
