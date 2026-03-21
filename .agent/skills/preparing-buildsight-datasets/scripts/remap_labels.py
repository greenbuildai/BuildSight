import os
import argparse

def remap_yolo_labels(labels_dir, mapping):
    """
    Remaps YOLO label IDs based on the mapping dictionary {old_id: new_id}.
    IDs not in mapping are discarded.
    """
    print(f"🔄 Remapping YOLO labels in {labels_dir}...")
    files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    
    for filename in files:
        path = os.path.join(labels_dir, filename)
        new_lines = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                old_id = int(parts[0])
                if old_id in mapping:
                    new_id = mapping[old_id]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
        
        with open(path, "w") as f:
            f.write("\n".join(new_lines) + "\n")

    print(f"✅ Remapped {len(files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildSight Label Remapper")
    parser.add_argument("--dir", required=True, help="Directory containing .txt labels")
    parser.add_argument("--source", required=True, choices=["ultralytics-ppe", "sh17"], help="Source dataset type")
    
    args = parser.parse_args()
    
    # Define Mappings to Protocol v1.0 (0: Person, 1: Helmet, 2: Vest)
    mappings = {
        "ultralytics-ppe": {
            6: 0, # Person
            0: 1, # Helmet
            2: 2  # Vest
        },
        "sh17": {
            0: 0,  # Person
            10: 1, # Helmet
            16: 2  # Vest
        }
    }
    
    remap_yolo_labels(args.dir, mappings[args.source])
