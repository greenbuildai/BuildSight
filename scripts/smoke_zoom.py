import cv2
import numpy as np
from pathlib import Path

src = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/smoke_test_before_after.jpg")
img = cv2.imread(str(src))

# Each row is 480px tall + 36px header
# Rows 3,4,5 (0-indexed) are the excavator/crane scenes
# synthetic images: rows 4,5,6,7,8,9
header_h = 36
row_h = 480

# Extract rows 3-7 (the interesting machinery scenes)
start = header_h + 3 * row_h
end   = header_h + 8 * row_h
crop  = img[start:end, :]

out = Path("e:/Company/Green Build AI/Prototypes/BuildSight/scripts/smoke_zoom.jpg")
cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"Saved -> {out}")
