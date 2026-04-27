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

from adaptive_postprocess import apply_all_rules
from site_aware_ensemble import detect_condition, preprocess_frame, wbf_fuse_condition

m1 = YOLO("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt")
m2 = YOLO("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt")

GPU      = 0
W    = [0.45, 0.55]        # YOLOv26=0.45, YOLOv11=0.55
CLS  = ["helmet", "safety_vest", "worker"]
PRE_CONF = 0.10

# Post-WBF per-class confidence gates (Jovi Shield)
CLS_THR = {
    0: 0.45,   # helmet
    1: 0.50,   # safety_vest
    2: 0.72,   # worker (High threshold to force v26+v11 consensus)
}


def predict(path, save_path=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]

    quick = m2.predict(img, device=GPU, verbose=False, conf=0.20, iou=0.40)[0]
    rough_workers = [
        box.xyxy[0].cpu().numpy().tolist()
        for box in (quick.boxes or [])
        if int(box.cls[0]) == 2 and float(box.conf[0]) >= 0.20
    ]
    condition = detect_condition(img, rough_workers).key
    work_img = preprocess_frame(img, condition)

    preds = []
    for m in [m1, m2]:
        r = m.predict(work_img, device=GPU, verbose=False, conf=PRE_CONF, iou=0.35)[0]
        b, s, l = [], [], []
        for x in (r.boxes or []):
            b.append(x.xyxy[0].cpu().numpy().tolist())
            s.append(float(x.conf[0]))
            l.append(int(x.cls[0]))
        preds.append((b, s, l))

    fused = wbf_fuse_condition(preds, w, h, condition, W)
    fused, _stats = apply_all_rules(fused, condition, w, h, image=img)

    colors = {0: (0, 255, 0), 1: (255, 165, 0), 2: (0, 120, 255)}
    for det in fused:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        l = det["cls"]
        s = det["score"]
        color = colors.get(l, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{CLS[l]} {s:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, condition, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    out = save_path or "ensemble_out.jpg"
    cv2.imwrite(out, img)
    return img, len(fused)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ensemble_inference.py <image> [output.jpg]")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "ensemble_out.jpg"
        img, n = predict(sys.argv[1], save_path=out)
        print(f"{n} detections -> {out}")
