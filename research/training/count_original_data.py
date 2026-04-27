import os

BASE_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset"

def get_counts():
    print(f"{'Folder Name':<25} | {'Original':<10} | {'Synthetic':<10} | {'Total':<10}")
    print("-" * 65)
    
    total_original = 0
    total_synthetic = 0
    
    # Define synthetic prefixes we use
    synthetic_markers = ['synthetic', 'syn_', 'synthetic_imagen_']
    
    folders = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    for folder in folders:
        folder_path = os.path.join(BASE_DIR, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        # Original: No 'synthetic' or 'syn_' in name
        original = [f for f in files if all(marker.lower() not in f.lower() for marker in synthetic_markers)]
        synthetic = [f for f in files if any(marker.lower() in f.lower() for marker in synthetic_markers)]
        
        print(f"{folder:<25} | {len(original):<10} | {len(synthetic):<10} | {len(files):<10}")
        
        total_original += len(original)
        total_synthetic += len(synthetic)
        
    print("-" * 65)
    print(f"{'TOTAL':<25} | {total_original:<10} | {total_synthetic:<10} | {total_original + total_synthetic:<10}")

if __name__ == "__main__":
    get_counts()
