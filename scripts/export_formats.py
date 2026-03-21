import os
import json
import csv
from tqdm import tqdm

OUTPUT_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset"
# Default to train split; pass --split val|test to override
SPLIT = os.environ.get("EXPORT_SPLIT", "train")
COCO_JSON = os.path.join(OUTPUT_DIR, "annotations", f"instances_{SPLIT}.json")
YOLO_BBOX_DIR = os.path.join(OUTPUT_DIR, "labels", SPLIT)
YOLO_SEG_DIR = os.path.join(OUTPUT_DIR, "labels_seg", SPLIT)
CSV_PATH = os.path.join(OUTPUT_DIR, f"scene_conditions_{SPLIT}.csv")

# Class ID mapping (locked schema — must match annotate_indian_dataset.py)
# 0=helmet, 1=safety_vest, 2=worker
# COCO category_id == YOLO class_id directly (no offset).
# The old cat_id-1 convention is NOT used here.

def convert_coco_to_yolo():
    print("Loading COCO JSON...")
    if not os.path.exists(COCO_JSON):
        print("COCO JSON not found! Run annotate_indian_dataset.py first.")
        return
        
    with open(COCO_JSON, 'r') as f:
        data = json.load(f)
        
    os.makedirs(YOLO_BBOX_DIR, exist_ok=True)
    os.makedirs(YOLO_SEG_DIR, exist_ok=True)
    
    images = {img['id']: img for img in data['images']}
    
    # Write global classification CSV
    print("Exporting Image Classifications (CSV)...")
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "scene_condition"])
        for img in data['images']:
            writer.writerow([img['file_name'], img.get('scene_condition', 'Unknown')])
            
    print("Exporting YOLO formats...")
    
    # We need to process image by image
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    for img_id, img_info in tqdm(images.items()):
        file_name = img_info['file_name']
        txt_name = os.path.splitext(file_name)[0] + ".txt"
        
        bbox_path = os.path.join(YOLO_BBOX_DIR, txt_name)
        seg_path = os.path.join(YOLO_SEG_DIR, txt_name)
        
        # If no annotations, write empty files (YOLO requirement for negative samples)
        if img_id not in img_to_anns:
            open(bbox_path, 'w').close()
            open(seg_path, 'w').close()
            continue
            
        w, h = img_info['width'], img_info['height']
        anns = img_to_anns[img_id]
        
        with open(bbox_path, 'w') as fb, open(seg_path, 'w') as fs:
            for ann in anns:
                # category_id IS the YOLO class_id directly (no offset).
                # Schema: 0=helmet, 1=safety_vest, 2=worker
                cat_id = ann['category_id']

                # 1. BBox Format: <class> <x_center> <y_center> <width> <height>
                bx, by, bw, bh = ann['bbox']  # COCO is [x, y, w, h] absolute

                cx = (bx + (bw / 2)) / w
                cy = (by + (bh / 2)) / h
                nw = bw / w
                nh = bh / h

                fb.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                # 2. Seg Format: <class> <x1> <y1> <x2> <y2> ...
                if 'segmentation' in ann and ann['segmentation']:
                    poly = ann['segmentation'][0]  # Take first polygon
                    normalized_poly = []

                    for i in range(0, len(poly) - 1, 2):
                        px, py = poly[i], poly[i + 1]
                        normalized_poly.append(f"{px / w:.6f}")
                        normalized_poly.append(f"{py / h:.6f}")

                    poly_str = " ".join(normalized_poly)
                    fs.write(f"{cat_id} {poly_str}\n")
                else:
                    # Fallback: write bbox as degenerate polygon
                    fs.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    
    print("\nConversion Complete!")
    print(f"- Standard YOLO labels saved to: {YOLO_BBOX_DIR}")
    print(f"- YOLO Polygon labels saved to: {YOLO_SEG_DIR}")
    print(f"- Scene classification saved to: {CSV_PATH}")

if __name__ == "__main__":
    convert_coco_to_yolo()
