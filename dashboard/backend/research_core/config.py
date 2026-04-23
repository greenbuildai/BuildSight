import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUTS_DIR = BASE_DIR / "inputs"
OUTPUTS_DIR = BASE_DIR / "output"
UPLOADS_DIR = INPUTS_DIR / "uploads"

# Create directories if they don't exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
MODEL_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\weights\yolov26_buildsight_best.pt"

# Default to the specific Trueview Camera IP provided
# Trying standard 'admin' username with provided password
DEFAULT_RTSP = "rtsp://admin:joseva%238765@192.168.43.100:554/stream1"
FILE_SOURCE = str(INPUTS_DIR / "ppe_video/PPE_1.mp4")

# Set VIDEO_SOURCE to None by default to prevent automatic camera connection on startup
# Can be overridden with environment variables
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE_RTSP") or os.getenv("VIDEO_SOURCE_FILE") or None

# Class Mapping
CLASS_NAMES = {
    0: "Person", 1: "Ear", 2: "Ear-mufs", 3: "Face", 4: "Face-guard", 5: "Face-mask",
    6: "Foot", 7: "Tool", 8: "Glasses", 9: "Gloves", 10: "Helmet", 11: "Hands", 12: "Head",
    13: "Medical-suit", 14: "Shoes", 15: "Safety-suit", 16: "Safety-vest"
}

# Server Configuration
HOST = "0.0.0.0"
PORT = 8000

# GIS Configuration
# Placeholder for red-zone coordinates (to be extracted from gis notebook)
RED_ZONE_COORDS = [] 
