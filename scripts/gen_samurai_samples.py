import sys, os, cv2, json
import torch
import numpy as np

# SAMURAI uses SAM2 backbone — check if available
sys.path.insert(0, '/nfsshare/joseva/SAMURAI')
os.makedirs('/nfsshare/joseva/annotated_samples/SAMURAI', exist_ok=True)

# Load test annotations to find images
with open('/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_test.json') as f:
    coco = json.load(f)

IMAGE_BASES = [
    '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/images/test',
    '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/test',
]

for cond in ['normal', 'dusty', 'low_light', 'crowded']:
    img_meta = next((x for x in coco['images'] if x.get('scene_condition') == cond), None)
    if img_meta is None:
        print(f'No image for {cond}'); continue

    img_path = None
    for base in IMAGE_BASES:
        p = os.path.join(base, img_meta['file_name'])
        if os.path.exists(p): img_path = p; break

    if img_path is None:
        print(f'Image not found for {cond}: {img_meta["file_name"]}'); continue

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to read {img_path}")
        continue

    # Add text overlay showing SAMURAI is a tracker
    out = frame.copy()
    cv2.putText(out, 'SAMURAI: Video Tracker (no single-frame inference)',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(out, f'Condition: {cond} | Composite Score: -0.2159',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

    out_path = f'/nfsshare/joseva/annotated_samples/SAMURAI/{cond}.jpg'
    cv2.imwrite(out_path, out)
    print(f'Saved: {out_path}')

print('SAMURAI annotated images done.')
