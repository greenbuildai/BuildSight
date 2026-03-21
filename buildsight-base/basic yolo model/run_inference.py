
import os
import cv2
from ultralytics import YOLO

def main():
    # Base paths
    BASE_DIR = r"e:\Company\Green Build AI\Prototypes\BuildSight\buildsight-base\basic yolo model"
    
    # Model Path - prioritize the trained 'best.pt'
    MODEL_PATH = os.path.join(BASE_DIR, "output", "kaggle_working_all_outputs", "kaggle", "working", "runs", "train", "weights", "best.pt")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Checking for 'yolov8n.pt'...")
        MODEL_PATH = os.path.join(BASE_DIR, "output", "yolov8n.pt")
        if not os.path.exists(MODEL_PATH):
             # Fallback to downloading
             print("yolov8n.pt not found locally. Using 'yolov8n.pt' (will download)...")
             MODEL_PATH = "yolov8n.pt"
    
    # Video Path
    INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "trial2.mp4")
    
    # Output Directory
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "predict_run_trial2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Input video: {INPUT_VIDEO_PATH}")
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Video file not found at {INPUT_VIDEO_PATH}")
        return

    print(f"Output directory: {OUTPUT_DIR}")

    # Run inference
    print("Starting inference on trial2.mp4...")
    # Using stream=True to process frame by frame helps with memory but let's use save=True for simplicity
    # exist_ok=True allows overwriting
    
    try:
        results = model.predict(
            source=INPUT_VIDEO_PATH, 
            save=True, 
            conf=0.25, 
            project=OUTPUT_DIR, 
            name="predict", 
            exist_ok=True
        )
        print(f"Inference complete. Results saved in {os.path.join(OUTPUT_DIR, 'predict')}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
