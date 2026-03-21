
import json
import os

nb_path = r"e:\Company\Green Build AI\Prototypes\BuildSight\buildsight-base\basic yolo model\code\ppe-detection.ipynb"

try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print("Notebook Cells Analysis:")
    for i, cell in enumerate(nb.get('cells', [])):
        cell_type = cell.get('cell_type')
        source = "".join(cell.get('source', []))
        if len(source) > 100:
            snippet = source[:100] + "..."
        else:
            snippet = source
        print(f"Cell {i} ({cell_type}): {snippet}")
        
        # Check for inference/prediction keywords
        if "predict" in source or "inference" in source or "YOLO" in source or "model(" in source:
             print(f"  >>> FOUND INTERESTING CODE: {source[:200]}")

except Exception as e:
    print(f"Error reading notebook: {e}")
