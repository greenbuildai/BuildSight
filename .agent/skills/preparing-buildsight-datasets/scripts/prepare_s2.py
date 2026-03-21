import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def convert_coco_to_yolo(json_path, img_dir, out_img_dir, out_label_dir, limit=500):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Map category_id to 0 (Person) as per protocol
    # In CHDataset, let's find which category is 'Person'
    cat_id_map = {}
    for cat in data['categories']:
        # CHDataset categories usually: 'hand', 'helmet', 'vest', 'person' or similar
        # Let's check categories
        # For CHDataset (S2), we usually want to map anything related to person to 0
        if 'person' in cat['name'].lower() or 'worker' in cat['name'].lower():
            cat_id_map[cat['id']] = 0
        elif 'helmet' in cat['name'].lower():
            cat_id_map[cat['id']] = 1
        elif 'vest' in cat['name'].lower():
            cat_id_map[cat['id']] = 2
    
    print(f"Mapping categories: {cat_id_map}")
    
    images = data['images'][:limit]
    img_id_to_file = {img['id']: img['file_name'] for img in images}
    img_data = {img['id']: img for img in images}
    
    annotations = [ann for ann in data['annotations'] if ann['image_id'] in img_id_to_file]
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    for img_id, file_name in tqdm(img_id_to_file.items(), desc="Processing S2"):
        # Copy and Enhance Image
        src_path = os.path.join(img_dir, file_name)
        if not os.path.exists(src_path):
            continue
            
        img = cv2.imread(src_path)
        if img is None:
            continue
            
        # Dehazing (Simple dark channel prior approximation or just contrast enhancement for dusty)
        # For dusty scenario, we use CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        cv2.imwrite(os.path.join(out_img_dir, file_name), enhanced)
        
        # Convert Annotations
        h, w = img_data[img_id]['height'], img_data[img_id]['width']
        relevant_anns = [ann for ann in annotations if ann['image_id'] == img_id]
        
        label_file = os.path.join(out_label_dir, os.path.splitext(file_name)[0] + '.txt')
        with open(label_file, 'w') as lf:
            for ann in relevant_anns:
                cat_id = ann['category_id']
                if cat_id not in cat_id_map:
                    continue
                
                protocol_id = cat_id_map[cat_id]
                bbox = ann['bbox'] # [x, y, w, h]
                
                x_center = (bbox[0] + bbox[2]/2) / w
                y_center = (bbox[1] + bbox[3]/2) / h
                width = bbox[2] / w
                height = bbox[3] / h
                
                lf.write(f"{protocol_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

if __name__ == "__main__":
    base_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\S2_dusty"
    json_path = os.path.join(base_dir, "temp_extracted", "train", "_annotations.coco.json")
    img_dir = os.path.join(base_dir, "temp_extracted", "train")
    out_img_dir = os.path.join(base_dir, "images")
    out_label_dir = os.path.join(base_dir, "labels_yolo")
    
    convert_coco_to_yolo(json_path, img_dir, out_img_dir, out_label_dir, limit=500)
