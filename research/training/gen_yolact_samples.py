import sys, os, cv2, json
import torch
import numpy as np

# Setup paths for YOLACT
YOLACT_DIR = '/nfsshare/joseva/yolact'
sys.path.insert(0, YOLACT_DIR)
sys.path.insert(0, os.path.join(YOLACT_DIR, 'external', 'DCNv2'))
os.chdir(YOLACT_DIR)

from yolact import Yolact
from data import cfg, set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess

def to_np(x):
    import torch
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)): return type(x)(to_np(v) for v in x)
    return np.array(x)

print("Loading YOLACT++ model...")
set_cfg('buildsight_config')
cfg.mask_proto_debug = False
net = Yolact()
net.load_weights('/nfsshare/joseva/yolact/weights/buildsight_207_80000.pth')
net.eval()
net.detect.use_fast_nms = True
net.detect.use_cross_class_nms = False
net = net.cuda(0)
torch.backends.cudnn.fastest = True

CLASS_NAMES = ['helmet', 'safety_vest', 'worker']
COLORS = {'helmet': (0,255,0), 'safety_vest': (255,165,0), 'worker': (0,120,255)}
SCORE_THR = 0.20

os.makedirs('/nfsshare/joseva/annotated_samples/YOLACT_plusplus', exist_ok=True)

# Load test annotations
with open('/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_test.json') as f:
    coco = json.load(f)

for cond in ['normal', 'dusty', 'low_light', 'crowded']:
    img_meta = next((x for x in coco['images'] if x.get('scene_condition') == cond), None)
    if img_meta is None:
        print(f'No image for {cond}'); continue

    img_path = None
    for base in [
        '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/images/test',
        '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/images',
        '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/test',
    ]:
        p = os.path.join(base, img_meta['file_name'])
        if os.path.exists(p):
            img_path = p; break

    if img_path is None:
        print(f'Image not found for {cond}: {img_meta["file_name"]}'); continue

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to read {img_path}")
        continue
    H, W = frame.shape[:2]

    with torch.no_grad():
        ft = torch.from_numpy(frame).cuda(0).float()
        b  = FastBaseTransform()(ft.unsqueeze(0))
        p  = net(b)
        cls_t, scr_t, box_t, _ = postprocess(p, W, H, score_threshold=SCORE_THR,
                                               visualize_lincomb=False, crop_masks=True)

    if len(cls_t) == 0:
        print(f'{cond}: no detections'); out = frame.copy()
    else:
        pc = to_np(cls_t).astype(int)
        ps = to_np(scr_t)
        pb = to_np(box_t).astype(int)
        out = frame.copy()
        for i in range(len(pc)):
            ci = pc[i]
            sc = ps[i]
            bx = pb[i]
            if sc < SCORE_THR: continue
            cn = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else str(ci)
            color = COLORS.get(cn, (200,200,200))
            x1,y1,x2,y2 = bx
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            label = f'{cn} {sc:.2f}'
            cv2.putText(out, label, (x1, max(y1-5,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = f'/nfsshare/joseva/annotated_samples/YOLACT_plusplus/{cond}.jpg'
    cv2.imwrite(out_path, out)
    print(f'Saved: {out_path}')

print('YOLACT++ annotated images done.')
