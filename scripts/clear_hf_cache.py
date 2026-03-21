import os
import shutil
from pathlib import Path

def clear_hf_cache():
    home_dir = Path.home()
    hf_cache_dir = home_dir / ".cache" / "huggingface" / "hub"
    
    print(f"Checking HuggingFace cache directory: {hf_cache_dir}")
    
    if not hf_cache_dir.exists():
        print("HuggingFace cache directory does not exist. Nothing to clear.")
        return
        
    total_size = 0
    directories_to_delete = []
    
    # Calculate size and find model directories
    for item in hf_cache_dir.iterdir():
        if item.is_dir() and item.name.startswith("models--"):
            item_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            total_size += item_size
            directories_to_delete.append(item)
            print(f"Found model: {item.name} (Size: {item_size / (1024**3):.2f} GB)")
            
    if not directories_to_delete:
        print("No models found in cache.")
        return
        
    print(f"Total space to be freed: {total_size / (1024**3):.2f} GB")
    
    # Delete the directories
    for directory in directories_to_delete:
        try:
            print(f"Deleting {directory.name}...")
            shutil.rmtree(directory)
            print(f"Successfully deleted {directory.name}")
        except Exception as e:
            print(f"Failed to delete {directory.name}. Error: {e}")
            
    print("Cleanup completed.")

if __name__ == "__main__":
    clear_hf_cache()
