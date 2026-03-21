import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

def prepare_final_dataset(base_dir, output_dir, split_ratio=0.8):
    # Definition of data sources
    scenarios = {
        'S1': os.path.join(base_dir, 'S1_normal'),
        'S2': os.path.join(base_dir, 'S2_dusty'),
        'S3': os.path.join(base_dir, 'S3_lowlight'),
        'S4': os.path.join(base_dir, 'S4_crowded')
    }
    
    # Output structure
    dirs = ['train/images', 'train/labels', 'val/images', 'val/labels']
    for d in dirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)
        
    stats = {'train': 0, 'val': 0}
    
    for scenario_name, scenario_path in scenarios.items():
        img_dir = os.path.join(scenario_path, 'images')
        label_dir = os.path.join(scenario_path, 'labels_yolo')
        
        if not os.path.exists(img_dir) or not os.path.exists(label_dir):
            print(f"Skipping {scenario_name}: directories not found.")
            continue
            
        # Get valid pairs
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        valid_pairs = []
        for img_file in images:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if os.path.exists(os.path.join(label_dir, label_file)):
                valid_pairs.append((img_file, label_file))
                
        # Shuffle and Split
        random.shuffle(valid_pairs)
        split_idx = int(len(valid_pairs) * split_ratio)
        train_set = valid_pairs[:split_idx]
        val_set = valid_pairs[split_idx:]
        
        print(f"Processing {scenario_name}: {len(train_set)} train, {len(val_set)} val")
        
        # Copy files
        for subset_name, subset_data in [('train', train_set), ('val', val_set)]:
            for img_f, lbl_f in tqdm(subset_data, desc=f"{scenario_name} {subset_name}"):
                # New filename to avoid collisions
                new_name = f"{scenario_name}_{img_f}"
                new_lbl_name = f"{scenario_name}_{lbl_f}"
                
                shutil.copy2(
                    os.path.join(img_dir, img_f),
                    os.path.join(output_dir, subset_name, 'images', new_name)
                )
                shutil.copy2(
                    os.path.join(label_dir, lbl_f),
                    os.path.join(output_dir, subset_name, 'labels', new_lbl_name)
                )
                stats[subset_name] += 1

    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'val/images', # Using val as test for prototype
        'nc': 3,
        'names': ['Person', 'Helmet', 'Vest']
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"Dataset creation complete. Total Train: {stats['train']}, Total Val: {stats['val']}")
    print(f"Data.yaml created at {os.path.join(output_dir, 'data.yaml')}")

if __name__ == "__main__":
    base_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset"
    output_dir = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_YOLOv8"
    prepare_final_dataset(base_dir, output_dir)
