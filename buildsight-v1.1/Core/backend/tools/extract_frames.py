import cv2
import os
from pathlib import Path

def extract_frames(video_path, output_dir, interval=30):
    """
    Extracts frames from a video file at a given interval.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    saved_count = 0
    
    print(f"Extracting frames from {video_path} to {output_dir}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % interval == 0:
            frame_name = f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Done. Extracted {saved_count} frames.")

if __name__ == "__main__":
    # Default configuration
    VIDEO_PATH = "inputs/ppe_video/PPE_1.mp4" # Adjust as needed
    OUTPUT_DIR = "training_data/images"
    
    # Check if we are running from root or backend
    if not os.path.exists(VIDEO_PATH) and os.path.exists(f"../{VIDEO_PATH}"):
         VIDEO_PATH = f"../{VIDEO_PATH}"
         OUTPUT_DIR = f"../{OUTPUT_DIR}"

    extract_frames(VIDEO_PATH, OUTPUT_DIR, interval=30)
