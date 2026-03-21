import os
import cv2
from tqdm import tqdm

DATA_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset"

# Folders to scan for videos
FOLDERS_TO_SCAN = ["Normal_Site_Condition", "Dusty_Condition", "Low_Light_Condition", "Crowded_Condition"]
SUPPORTED_VIDEOS = ('.mp4', '.avi', '.mov', '.mkv')
FPS_EXTRACT = 2 # Extract 2 frames per second of video

def extract_frames():
    total_videos = 0
    total_frames = 0
    
    for folder in FOLDERS_TO_SCAN:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path): continue
            
        videos = [f for f in os.listdir(folder_path) if f.lower().endswith(SUPPORTED_VIDEOS)]
        
        for video_name in videos:
            total_videos += 1
            video_path = os.path.join(folder_path, video_name)
            
            # Setup video capture
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps == 0:
                print(f"[{folder}] Warning: Could not read FPS from {video_name}. Skipping.")
                continue
                
            frame_interval = int(fps / FPS_EXTRACT)
            if frame_interval < 1: frame_interval = 1
            
            print(f"[{folder}] Extracting frames from {video_name} (FPS: {fps}, Interval: {frame_interval})...")
            
            frame_count = 0
            extracted = 0
            base_name = os.path.splitext(video_name)[0]
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_count % frame_interval == 0:
                    out_name = f"video_frame_{base_name}_{extracted}.jpg"
                    out_path = os.path.join(folder_path, out_name)
                    
                    cv2.imwrite(out_path, frame)
                    extracted += 1
                    total_frames += 1
                    
                frame_count += 1
                
            cap.release()
            print(f"  -> Extracted {extracted} frames.")
            
    print(f"\nExtraction Complete! Found {total_videos} videos and extracted {total_frames} frames.")

if __name__ == "__main__":
    extract_frames()
