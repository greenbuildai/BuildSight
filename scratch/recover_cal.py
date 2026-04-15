import cv2
import numpy as np
import json
from datetime import datetime

# Recovered from user terminal output
PX_PTS = np.float32([[844, 466], [756, 61], [32, 71], [1, 440]])
WD_PTS = np.float32([[0.0, 0.0], [0.0, 9.75], [18.9, 9.75], [18.9, 0.0]])

# Compute Homography
H, _ = cv2.findHomography(PX_PTS, WD_PTS)

# Save the files
H_PATH = "camera_cam01_H.npy"
META_PATH = "camera_cam01_meta.json"

np.save(H_PATH, H)

meta = {
    "camera_id": "CAM-01",
    "calibrated_at": datetime.now().isoformat(),
    "frame_size": [848, 478], # From log
    "pixel_points": {
        "SW": [844.0, 466.0],
        "SE": [756.0, 61.0],
        "NE": [32.0, 71.0],
        "NW": [1.0, 440.0]
    },
    "world_points": {
        "SW": [0.0, 0.0],
        "SE": [0.0, 9.75],
        "NE": [18.9, 9.75],
        "NW": [18.9, 0.0]
    },
    "mean_reprojection_error_m": 0.0,
    "H_matrix": H.tolist(),
    "site": {
        "sw_corner": "10.816539, 78.668835",
        "width_m": 18.9,
        "depth_m": 9.75
    }
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("SUCCESS: Recovered camera_cam01_H.npy and camera_cam01_meta.json")
