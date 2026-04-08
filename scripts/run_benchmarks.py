import torch, time, json, cv2, glob, os
import numpy as np
from ultralytics import YOLO

GPU = 0
results = {}

test_imgs = glob.glob('/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/test/*.jpg')[:100]
print(f'Using {len(test_imgs)} test images')

for model_name, weights in [
    ('YOLOv11', '/nfsshare/joseva/buildsight/runs/detect/yolov11_buildsight/weights/best.pt'),
    ('YOLOv26', '/nfsshare/joseva/buildsight/runs/detect/yolov26_buildsight/weights/best.pt'),
]:
    if not os.path.exists(weights):
        print(f"Skipping {model_name}: weights not found at {weights}")
        continue
    model = YOLO(weights)
    torch.cuda.reset_peak_memory_stats(GPU)
    t0 = time.time()
    for p in test_imgs:
        model(p, device=f'cuda:{GPU}', verbose=False)
    elapsed = time.time() - t0
    results[model_name] = {
        'peak_vram_gb': round(torch.cuda.max_memory_allocated(GPU)/1e9, 2),
        'fps': round(len(test_imgs)/elapsed, 1)
    }
    print(f'{model_name}: {results[model_name]}')
    del model; torch.cuda.empty_cache()

# YOLACT++ and SAMURAI estimates based on eval
results['YOLACT_plusplus'] = {'peak_vram_gb': 9.2, 'fps': 13.9, 'note': 'FPS from eval lat'}
results['SAMURAI'] = {'peak_vram_gb': 12.0, 'fps': 4.0, 'note': 'Video tracker estimate'}

with open('/nfsshare/joseva/vram_fps_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved to /nfsshare/joseva/vram_fps_results.json')
print(json.dumps(results, indent=2))
