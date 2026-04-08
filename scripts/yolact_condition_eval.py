#!/usr/bin/env python3
"""
YOLACT++ Per-Condition Evaluation - BuildSight Phase 2
Evaluates buildsight_207_80000.pth on each site condition (S1-S4)
Output: /nfsshare/joseva/yolact_condition_eval.json
"""
import sys, os, json, time
import numpy as np
from collections import defaultdict

YOLACT_DIR = '/nfsshare/joseva/yolact'
sys.path.insert(0, YOLACT_DIR)
sys.path.insert(0, os.path.join(YOLACT_DIR, 'external', 'DCNv2'))
os.chdir(YOLACT_DIR)

import torch, cv2
from yolact import Yolact
from data import cfg, set_cfg
from utils.augmentations import FastBaseTransform
from layers.output_utils import postprocess

WEIGHTS  = '/nfsshare/joseva/yolact/weights/buildsight_207_80000.pth'
ANN_PATH = '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/annotations/instances_test.json'
OUT_PATH = '/nfsshare/joseva/yolact_condition_eval.json'
SCORE_THR = 0.15
TOP_K     = 100
GPU_ID    = int(os.environ.get('YOLACT_GPU_ID', '1'))
IMG_DIR_CANDIDATES = [
    '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLACT_plusplus/images',
    '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/test',
    '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/images/test',
    '/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/SAMURAI/images/test',
]

CONDITIONS  = {'S1': 'normal', 'S2': 'dusty', 'S3': 'low_light', 'S4': 'crowded'}
CLASS_NAMES = ['helmet', 'safety_vest', 'worker']


def to_np(x):
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            if hasattr(item, 'detach'):
                out.append(item.detach().cpu().numpy())
            else:
                out.append(item)
        return np.array(out)
    return np.array(x)


def resolve_image_path(file_name):
    for img_dir in IMG_DIR_CANDIDATES:
        candidate = os.path.join(img_dir, file_name)
        if os.path.exists(candidate):
            return candidate
    return None


def compute_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter, 1e-6)


def ap11(tp_arr, fp_arr, n_gt):
    if n_gt == 0:
        return 0.0
    tp_c = np.cumsum(tp_arr)
    fp_c = np.cumsum(fp_arr)
    prec = tp_c / (tp_c + fp_c + 1e-10)
    rec  = tp_c / n_gt
    return sum(
        np.max(prec[rec >= t]) if (rec >= t).any() else 0
        for t in np.arange(0, 1.1, 0.1)
    ) / 11


print("[1/3] Loading YOLACT++ model...", flush=True)
set_cfg('buildsight_config')
cfg.mask_proto_debug = False
net = Yolact()
net.load_weights(WEIGHTS)
net.eval()
net.detect.use_fast_nms = True
net.detect.use_cross_class_nms = False
net = net.cuda(GPU_ID)
torch.backends.cudnn.fastest = True
print("    Model loaded OK.", flush=True)

print("[2/3] Loading annotations...", flush=True)
with open(ANN_PATH) as f:
    coco = json.load(f)
cat_map = {c['id']: i for i, c in enumerate(coco['categories'])}
print(f"    Categories: {[(c['id'], c['name']) for c in coco['categories']]}", flush=True)
print(f"    Total test images: {len(coco['images'])}", flush=True)

results = {}
print("[3/3] Evaluating per condition...", flush=True)

for s, cond in CONDITIONS.items():
    print(f"\n--- {s} ({cond}) ---", flush=True)
    imgs = [x for x in coco['images'] if x.get('scene_condition') == cond]
    iids = {x['id'] for x in imgs}
    anns = [a for a in coco['annotations'] if a['image_id'] in iids]
    print(f"  images={len(imgs)}  annotations={len(anns)}", flush=True)

    gt_ic = defaultdict(list)
    for a in anns:
        ci = cat_map.get(a['category_id'], -1)
        if ci < 0:
            continue
        x, y, w, h = a['bbox']
        gt_ic[(a['image_id'], ci)].append([x, y, x + w, y + h])

    n_gt_cls = defaultdict(int)
    for (iid, ci), bs in gt_ic.items():
        n_gt_cls[ci] += len(bs)

    dets   = defaultdict(list)
    tp_all = fp_all = fn_all = 0
    lats   = []

    for idx, meta in enumerate(imgs):
        fp_img = resolve_image_path(meta['file_name'])
        if fp_img is None:
            continue
        frame  = cv2.imread(fp_img)
        if frame is None:
            continue
        H, W = frame.shape[:2]

        t0 = time.time()
        with torch.no_grad():
            ft = torch.from_numpy(frame).cuda(GPU_ID).float()
            b  = FastBaseTransform()(ft.unsqueeze(0))
            p  = net(b)
        with torch.no_grad():
            cls_t, scr_t, box_t, _ = postprocess(
                p, W, H,
                score_threshold=SCORE_THR,
                visualize_lincomb=False,
                crop_masks=True
            )
        lats.append((time.time() - t0) * 1000)

        if len(cls_t) == 0:
            for ci in range(len(CLASS_NAMES)):
                fn_all += len(gt_ic.get((meta['id'], ci), []))
            continue

        pc = to_np(cls_t)
        ps = to_np(scr_t[0] if isinstance(scr_t, (list, tuple)) else scr_t)
        pb = to_np(box_t)

        for ci in range(len(CLASS_NAMES)):
            gbs  = gt_ic.get((meta['id'], ci), [])
            mask = pc == ci
            cs   = ps[mask]
            cb   = pb[mask]
            used = [False] * len(gbs)
            for sc, bx in sorted(zip(cs, cb), key=lambda x: -x[0]):
                best_iou, best_j = 0, -1
                for j, gb in enumerate(gbs):
                    if used[j]:
                        continue
                    v = compute_iou(bx.tolist(), gb)
                    if v > best_iou:
                        best_iou = v
                        best_j   = j
                if best_iou >= 0.5 and best_j >= 0:
                    dets[ci].append((sc, 1, 0))
                    used[best_j] = True
                    tp_all += 1
                else:
                    dets[ci].append((sc, 0, 1))
                    fp_all += 1
            fn_all += sum(1 for u in used if not u)

        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{len(imgs)}]", flush=True)

    P   = tp_all / max(tp_all + fp_all, 1)
    R   = tp_all / max(tp_all + fn_all, 1)
    F1  = 2 * P * R / max(P + R, 1e-6)
    lat = float(np.mean(lats)) if lats else 0

    per_ap = {}
    for ci, cn in enumerate(CLASS_NAMES):
        if ci not in dets:
            per_ap[cn] = 0.0
            continue
        ds = sorted(dets[ci], key=lambda x: -x[0])
        per_ap[cn] = round(
            ap11(np.array([d[1] for d in ds]),
                 np.array([d[2] for d in ds]),
                 n_gt_cls[ci]), 4
        )
    mAP50 = float(np.mean(list(per_ap.values())))

    results[s] = {
        'images_evaluated': len(lats),
        'precision':        round(P, 4),
        'recall':           round(R, 4),
        'f1':               round(F1, 4),
        'mAP50':            round(mAP50, 4),
        'tp_total':         int(tp_all),
        'fp_total':         int(fp_all),
        'fn_total':         int(fn_all),
        'avg_lat_ms':       round(lat, 2),
        'per_class_ap50':   per_ap,
    }
    print(f"  RESULT {s}: P={P:.4f} R={R:.4f} F1={F1:.4f} mAP50={mAP50:.4f} lat={lat:.1f}ms", flush=True)

with open(OUT_PATH, 'w') as f:
    json.dump({'YOLACT_plusplus': results}, f, indent=2)

print(f"\n[DONE] Results saved to {OUT_PATH}", flush=True)
print(json.dumps({'YOLACT_plusplus': results}, indent=2), flush=True)
