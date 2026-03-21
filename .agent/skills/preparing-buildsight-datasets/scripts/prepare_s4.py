import os
import cv2
from tqdm import tqdm

def convert_widerperson_to_yolo(ann_dir, img_dir, out_img_dir, out_label_dir, limit=800):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)
    
    ann_files = os.listdir(ann_dir)
    count = 0
    
    for ann_file in tqdm(ann_files, desc="Processing S4"):
        if count >= limit:
            break
            
        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            lines = f.readlines()
            
        num_objects = int(lines[0].strip())
        objects = []
        person_count = 0
        
        for line in lines[1:]:
            parts = line.strip().split()
            if not parts: continue
            label = int(parts[0])
            # Labels 1, 2, 3 are persons
            if label in [1, 2, 3]:
                person_count += 1
                objects.append(parts)
        
        # S4 condition: >= 5 people
        if person_count < 5:
            continue
            
        img_name = ann_file.replace('.txt', '')
        src_img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(src_img_path):
            continue
            
        img = cv2.imread(src_img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Save image
        cv2.imwrite(os.path.join(out_img_dir, img_name), img)
        
        # Save label
        out_label_path = os.path.join(out_label_dir, img_name.replace('.jpg', '.txt'))
        with open(out_label_path, 'w') as lf:
            for obj in objects:
                # obj: [label, x1, y1, x2, y2]
                x1, y1, x2, y2 = map(float, obj[1:])
                
                # YOLO format: class x_center y_center width height (normalized)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                lf.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        count += 1

if __name__ == "__main__":
    base_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\S4_crowded"
    ann_dir = os.path.join(base_dir, "temp_extracted_wider", "Annotations")
    img_dir = os.path.join(base_dir, "temp_extracted_wider", "Images")
    out_img_dir = os.path.join(base_dir, "images")
    out_label_dir = os.path.join(base_dir, "labels_yolo")
    
    convert_widerperson_to_yolo(ann_dir, img_dir, out_img_dir, out_label_dir, limit=800)
