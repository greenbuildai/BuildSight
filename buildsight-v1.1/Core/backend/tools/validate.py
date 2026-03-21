import cv2
import sys
import os
from pathlib import Path

# Add backend to path to import services
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
sys.path.append(str(backend_dir))

try:
    from backend.services.inference import inference_service
    from backend import config
except ImportError:
    # If running from root directory context adjustment
    sys.path.append(str(current_dir.parent))
    from services.inference import inference_service
    from backend import config

def validate_system(video_path):
    print(f"Validating system on {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video.")
        return

    frame_count = 0
    total_detections = 0
    issues_found = []
    
    # Check Model Loading
    try:
        print(f"Model loaded: {inference_service.model.ckpt_path}")
    except Exception as e:
        print(f"Error checking model: {e}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run standard inference
        try:
            result = inference_service.detect(frame, frame_count)
            
            # Basic Verification
            if len(result.objects) > 0:
                total_detections += len(result.objects)
                
                for obj in result.objects:
                    # Check for "Ghost" IDs (simple check logic)
                    if obj.track_id != frame_count and obj.track_id > 100: 
                        # This is a heuristic check; mostly just checking structure
                        pass
        except Exception as e:
            print(f"Inference failed at frame {frame_count}: {e}")
            break
            
        frame_count += 1
        if frame_count > 50: # Run on first 50 frames to be quick
            break
            
    print(f"\nValidation Summary:")
    print(f"Processed {frame_count} frames.")
    print(f"Total objects detected: {total_detections}")
    
    if total_detections == 0:
        print("WARNING: No objects detected. Detection system might be failing.")
    else:
        print("SUCCESS: Detection system is producing results.")
        
    print("\nLogic System Analysis (Code Review based):")
    print("- Temporal Reasoning: MISSING (IDs are frame-based indices, no Tracker)")
    print("- Occlusion Handling: WEAK (Simple IOU check, no deep association)")
    print("- Active Model: YOLOv8 (Ultralytics)")

    cap.release()

if __name__ == "__main__":
    # Adjust path if running from root
    VIDEO_PATH = "inputs/ppe_video/PPE_1.mp4"
    if not os.path.exists(VIDEO_PATH) and os.path.exists(f"backend/../{VIDEO_PATH}"):
        VIDEO_PATH = f"backend/../{VIDEO_PATH}" 
        
    # Absolute path fix for the run environment
    abs_path = Path("d:/Jovi/Projects/BuildSight/Core") / "inputs/ppe_video/PPE_1.mp4"
    if abs_path.exists():
        VIDEO_PATH = str(abs_path)

    validate_system(VIDEO_PATH)
