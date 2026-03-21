import cv2
import os
import glob
from pathlib import Path

# Paths
BASE_DIR = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset")
OUTPUT_DIR = BASE_DIR / "Extracted_Video_Frames"

def extract_frames():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    video_files = glob.glob(str(BASE_DIR / "**" / "*.mp4"), recursive=True)
    
    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} video files.")
    
    frame_count = 0
    for video_path in video_files:
        print(f"Processing: {os.path.basename(video_path)}")
        vidcap = cv2.VideoCapture(video_path)
        fps = round(vidcap.get(cv2.CAP_PROP_FPS))
        
        # If unable to get valid fps, default to 30
        if fps <= 0:
            fps = 30
            
        success, image = vidcap.read()
        count = 0
        
        while success:
            # Save 2 frames per second
            if count % max(1, fps // 2) == 0:
                frame_idx = count / fps
                output_name = f"{Path(video_path).stem}_sec{frame_idx:.1f}.jpg"
                output_file = OUTPUT_DIR / output_name
                
                # Check if file exists to avoid overwriting unnecessarily
                if not output_file.exists():
                    cv2.imwrite(str(output_file), image)
                    frame_count += 1
            success, image = vidcap.read()
            count += 1
            
        vidcap.release()
    
    print(f"Extraction complete. Saved {frame_count} new frames to {OUTPUT_DIR.name}")

if __name__ == "__main__":
    extract_frames()
