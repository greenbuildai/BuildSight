#!/usr/bin/env python3
"""BuildSight Ensemble — WBF (YOLOv26 + YOLOv11)
Fixes applied 2026-03-31 (Toni):
  - Pre-prediction conf=0.30 to suppress weak single-model detections
  - WBF IoU threshold raised 0.50 -> 0.55 for tighter box consensus
  - Post-WBF per-class confidence gates:
      worker      >= 0.28  (most reliably detected)
      helmet      >= 0.32  (shape-distinctive, needs some confidence)
      safety_vest >= 0.38  (most FP-prone: dog/tarp/clothing killer)
"""
import cv2
import numpy as np
from ultralytics import YOLO

m1 = YOLO("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt")
m2 = YOLO("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt")

GPU      = 0
W    = [0.45, 0.55]        # YOLOv26=0.45, YOLOv11=0.55
CLS  = ["helmet", "safety_vest", "worker"]
PRE_CONF = 0.45
WBF_IOU  = 0.40

# Post-WBF per-class confidence gates (Jovi Shield)
CLS_THR = {
    0: 0.45,   # helmet
    1: 0.50,   # safety_vest
    2: 0.72,   # worker (High threshold to force v26+v11 consensus)
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)


def fuse(preds, weights, thr=WBF_IOU):
    ab, al_s, al_l = [], [], []
    for w, (b, s, l) in zip(weights, preds):
        for bi, si, li in zip(b, s, l):
            ab.append(bi)
            al_s.append(si)  # Removed score * weight to keep raw confidence for gates
            al_l.append(li)
    if not ab:
        return [], [], []

    order = np.argsort(-np.array(al_s))
    ab  = [ab[i]  for i in order]
    al_s = [al_s[i] for i in order]
    al_l = [al_l[i] for i in order]

    used, fb, fs, fl = set(), [], [], []
    for i in range(len(ab)):
        if i in used:
            continue
        cb, cs = [ab[i]], [al_s[i]]
        used.add(i)
        for j in range(i + 1, len(ab)):
            if j in used or al_l[i] != al_l[j]:
                continue
            if iou(ab[i], ab[j]) > thr:
                cb.append(ab[j])
                cs.append(al_s[j])
                used.add(j)

        ws = np.array(cs)
        ws = ws / ws.sum()
        fused_score = float(np.mean(cs))
        fused_cls   = al_l[i]

        # Post-WBF per-class confidence gate
        if fused_score >= CLS_THR.get(fused_cls, 0.30):
            box = np.average(cb, axis=0, weights=ws).tolist()
            
            # --- Jovi's Geometric Guardrails ---
            bx_w, bx_h = box[2] - box[0], box[3] - box[1]
            aspect_ratio = bx_w / max(bx_h, 1e-6)
            
            # 1. Human Aspect Ratio Check (Workers are tall: H > W)
            if fused_cls == 2 and aspect_ratio > 1.0:
                continue # Discard horizontal objects detected as workers (Excavators)
                
            # 2. Size Check (Workers rarely fill more than 15% of high-res image)
            if fused_cls == 2 and (bx_w * bx_h) > 0.15:
                continue
                
            fb.append(box)
            fs.append(fused_score)
            fl.append(fused_cls)

    return fb, fs, fl


def predict(path, save_path=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]

    preds = []
    for m in [m1, m2]:
        r = m.predict(path, device=GPU, verbose=False, conf=PRE_CONF)[0]
        b, s, l = [], [], []
        for x in (r.boxes or []):
            xy = x.xyxy[0].cpu().numpy().tolist()
            b.append([xy[0]/w, xy[1]/h, xy[2]/w, xy[3]/h])
            s.append(float(x.conf[0]))
            l.append(int(x.cls[0]))
        preds.append((b, s, l))

    fb, fs, fl = fuse(preds, W)

    colors = {0: (0, 255, 0), 1: (255, 165, 0), 2: (0, 120, 255)}
    for b, s, l in zip(fb, fs, fl):
        x1, y1 = int(b[0]*w), int(b[1]*h)
        x2, y2 = int(b[2]*w), int(b[3]*h)
        color = colors.get(l, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{CLS[l]} {s:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out = save_path or "ensemble_out.jpg"
    cv2.imwrite(out, img)
    return img, len(fb)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ensemble_inference.py <image> [output.jpg]")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "ensemble_out.jpg"
        img, n = predict(sys.argv[1], save_path=out)
        print(f"{n} detections -> {out}")
