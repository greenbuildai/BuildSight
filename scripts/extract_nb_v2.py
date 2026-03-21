
import json
import os

notebook_path = r'e:\Company\Green Build AI\Prototypes\BuildSight\buildsight_v0\basic yolo model\code\ppe-detection.ipynb'
output_path = r'e:\Company\Green Build AI\Prototypes\BuildSight\extract_results.txt'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(output_path, 'w', encoding='utf-8') as f_out:
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell.get('source', []))
        if 'predict' in source.lower() or 'validation' in source.lower() or 'best.pt' in source.lower():
            f_out.write(f"--- Cell {i} ({cell['cell_type']}) ---\n")
            f_out.write(source + "\n")
            if cell['cell_type'] == 'code':
                for output in cell.get('outputs', []):
                    if output['output_type'] == 'stream':
                        f_out.write(output['text'] + "\n")
                    elif output['output_type'] == 'execute_result':
                        f_out.write(str(output['data'].get('text/plain', '')) + "\n")
            f_out.write("-" * 20 + "\n")
