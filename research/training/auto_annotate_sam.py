import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T

# Segment Anything
from segment_anything import sam_model_registry, SamPredictor

# Paths - Update these to your local model weights!
DINO_CONFIG_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\weights\groundingdino_swint_ogc.pth"
SAM_WEIGHTS_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\weights\sam_vit_h_4b8939.pth"
DATA_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset"
OUTPUT_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset"

# Detection Prompt
TEXT_PROMPT = "person . safety helmet . safety vest ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Setup Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Models
print("Loading GroundingDINO...")
dino_model = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH, device=device)

print("Loading Segment Anything Model (SAM)...")
sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS_PATH).to(device=device)
sam_predictor = SamPredictor(sam)

# Categories
CATEGORIES = [
    {"supercategory": "person", "id": 1, "name": "person"},
    {"supercategory": "ppe", "id": 2, "name": "helmet"},
    {"supercategory": "ppe", "id": 3, "name": "safety_vest"}
]

# Mapping GroundingDINO phrases to our categories
CAT_MAP = {
    "person": 1,
    "safety helmet": 2,
    "safety vest": 3
}

def annotate_dataset():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    
    # Initialize COCO structure
    coco_data = {
        "info": {"description": "BuildSight Indian Construction Worker Dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES
    }
    
    annotation_id = 1
    image_id_counter = 1
    
    site_conditions = ["Normal_Site_Condition", "Low_Light_Condition", "Dusty_Condition", "Crowded_Condition"]
    
    # Process each site condition folder
    for condition in site_conditions:
        folder_path = os.path.join(DATA_DIR, condition)
        if not os.path.exists(folder_path):
            continue
            
        print(f"\nProcessing {condition}...")
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files):
            img_path = os.path.join(folder_path, img_name)
            
            # Use GroundingDINO's image loader
            image_source, image_tensor = load_image(img_path)
            h, w, _ = image_source.shape
            
            # Predict Bounding Boxes (GroundingDINO)
            boxes, logits, phrases = predict(
                model=dino_model,
                image=image_tensor,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device=device
            )
            
            # If nothing detected, skip annotation but keep image record
            has_annotations = len(boxes) > 0
            
            # Copy image to output folder
            new_img_name = f"{condition}_{img_name}"
            out_img_path = os.path.join(OUTPUT_DIR, "images", new_img_name)
            
            if not os.path.exists(out_img_path):
                 cv2.imwrite(out_img_path, cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR))
            
            # Add to COCO images list
            coco_data["images"].append({
                "id": image_id_counter,
                "file_name": new_img_name,
                "height": h,
                "width": w,
                "scene_condition": condition # Custom field for YOLACT
            })
            
            if has_annotations:
                # Prepare SAM Predictor
                sam_predictor.set_image(image_source)
                
                # GroundingDINO outputs normalized [cx, cy, w, h] - convert to absolute [x1, y1, x2, y2]
                xyxy_boxes = boxes * torch.Tensor([w, h, w, h])
                xyxy_boxes[:, :2] -= xyxy_boxes[:, 2:] / 2  # cx, cy to x1, y1
                xyxy_boxes[:, 2:] += xyxy_boxes[:, :2]      # w, h param to x2, y2
                
                for i in range(len(xyxy_boxes)):
                    phrase = phrases[i]
                    # Map phrase to category ID. Skip unknown detections.
                    cat_id = None
                    for key in CAT_MAP:
                        if key in phrase:
                            cat_id = CAT_MAP[key]
                            break
                    
                    if cat_id is None: continue
                    
                    box = xyxy_boxes[i].cpu().numpy()
                    score = logits[i].item()
                    
                    # Convert [x1, y1, x2, y2] to COCO format [x_min, y_min, width, height]
                    coco_box = [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])]
                    
                    # Run SAM using the bounding box as a prompt
                    masks, _, _ = sam_predictor.predict(
                        box=box,
                        multimask_output=False
                    )
                    
                    mask = masks[0]
                    # Convert boolean mask to polygons
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    polygons = []
                    for contour in contours:
                        # Simplify contour slightly
                        epsilon = 0.005 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        if len(approx) >= 3:
                            polygons.append(approx.flatten().tolist())
                            
                    # Calculate Area
                    area = float(np.sum(mask))
                    
                    if len(polygons) > 0:
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id_counter,
                            "category_id": cat_id,
                            "segmentation": polygons,
                            "area": area,
                            "bbox": coco_box,
                            "iscrowd": 0,
                            "score": score # Optional confidence tracking
                        })
                        annotation_id += 1
                        
            image_id_counter += 1
            
    # Save the giant COCO JSON
    json_path = os.path.join(OUTPUT_DIR, "instances_default.json")
    with open(json_path, 'w') as f:
        json.dump(coco_data, f)
        
    print(f"\nAuto-Annotation Complete! Generated {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")
    print("Saved to:", json_path)

if __name__ == "__main__":
    annotate_dataset()
