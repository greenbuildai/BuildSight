
import json
import os

notebook_path = r'e:\Company\Green Build AI\Prototypes\BuildSight\buildsight_v0\basic yolo model\code\ppe-detection.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    if any(s in source for s in ['S1', 'S2', 'S3', 'S4', 'Dust', 'Low-Light', 'Night', 'Crowd']):
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        print(source)
        found = True

if not found:
    print("None of the terms found in notebook cells.")
