import os
import argparse
from ultralytics import YOLO

def auto_annotate(source_dir, output_dir, model_path="yolov8n.pt", scenario="S1"):
    """
    Automates pre-annotation of images using YOLOv8.
    Protocol v1.0 Class Mapping: 0: Person, 1: Helmet, 2: Vest
    Note: COCO-trained YOLOv8n only detects 'Person' (ID 0).
    """
    print(f"🚀 Initializing Protocol-Aware Auto-Annotation for Scenario: {scenario}")
    
    # Load Model
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process Images
    # Scenario-specific thresholds for safety-critical sensitivity
    conf = 0.25
    if scenario == "S3": conf = 0.15 # Higher sensitivity for low-light
    if scenario == "S4": conf = 0.20 # Capture dense crowds
    
    results = model.predict(source=source_dir, save_txt=True, project=output_dir, name=scenario, conf=conf, classes=[0])
    
    # Note: Only Class 0 (Person) is predicted by default YOLOv8n.
    # Helmet (1) and Vest (2) must be manually refined in Phase 2.
    
    print(f"✅ Auto-Annotation complete for Class 0 (Person).")
    print(f"📍 Labels saved to {output_dir}/{scenario}/labels")
    if scenario == "S4":
        print(f"⚠️  REMINDER: S4 requires ≥ 5 workers. Validation script needed next.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildSight Auto-Annotation Script")
    parser.epilog = "Ensure 'ultralytics' is installed: pip install ultralytics"
    parser.add_argument("--source", required=True, help="Path to raw image directory")
    parser.add_argument("--output", default="Dataset/Draft_Labels", help="Path to save draft labels")
    parser.add_argument("--scenario", default="S1", choices=["S1", "S2", "S3", "S4"], help="BuildSight Scenario")
    
    args = parser.parse_args()
    auto_annotate(args.source, args.output, scenario=args.scenario)
