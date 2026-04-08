#!/usr/bin/env python3
"""
rerun_fixed_inference.py
========================
Re-runs YOLOv11 and YOLOv26 on the original val images using a precomputed
condition split and applies three inference-time fixes without retraining:

  FIX-A  Aspect ratio: worker boxes wider than tall (ratio > 1.25) are removed
  FIX-B  Cross-class NMS: overlapping cross-class boxes are suppressed
  FIX-C  Per-class confidence thresholds are raised vs the flat 0.25 default
"""

import json
from pathlib import Path

import cv2

MODEL_PATHS = {
    "yolo11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
}

CONDITION_SPLIT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")
VAL_IMG_DIR = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val")
OUT_DIR = Path("/nfsshare/joseva/val_annotated_fixed")

CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]

CLS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}
CLS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 100, 0)}
CONF_PER_CLASS = {0: 0.30, 1: 0.38, 2: 0.42}
CROSS_CLASS_NMS_IOU = 0.45
MAX_WORKER_ASPECT = 1.25
GPU = 0


def iou(box_a, box_b):
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - inter
    )
    return inter / max(area_union, 1e-6)


def apply_fixes(raw_boxes):
    boxes = [box for box in raw_boxes if box["score"] >= CONF_PER_CLASS[box["cls"]]]

    filtered = []
    for box in boxes:
        if box["cls"] == 2:
            x1, y1, x2, y2 = box["box"]
            width = x2 - x1
            height = y2 - y1
            if height > 0 and (width / height) > MAX_WORKER_ASPECT:
                continue
        filtered.append(box)

    boxes = sorted(filtered, key=lambda item: -item["score"])
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j] or boxes[i]["cls"] == boxes[j]["cls"]:
                continue
            if iou(boxes[i]["box"], boxes[j]["box"]) > CROSS_CLASS_NMS_IOU:
                keep[j] = False

    return [box for box, is_kept in zip(boxes, keep) if is_kept]


def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["box"]]
        color = CLS_COLORS[box["cls"]]
        label = f"{CLS_NAMES[box['cls']]} {box['score']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w + 2, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    return image


def run():
    from ultralytics import YOLO

    if not CONDITION_SPLIT_JSON.exists():
        print(f"ERROR: {CONDITION_SPLIT_JSON} not found")
        return

    splits = json.loads(CONDITION_SPLIT_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_key, model_path in MODEL_PATHS.items():
        model_label = "YOLOv11" if model_key == "yolo11" else "YOLOv26"
        print(f"\n=== {model_label} ===")
        model = YOLO(model_path)

        for condition in CONDITIONS:
            file_names = splits.get(condition, [])
            if not file_names:
                print(f"  [SKIP] No files for {condition}")
                continue

            dst_dir = OUT_DIR / condition / model_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {condition}: {len(file_names)} images -> {dst_dir}")

            total_before = 0
            total_after = 0

            for file_name in file_names:
                img_path = VAL_IMG_DIR / file_name
                if not img_path.exists():
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                result = model.predict(str(img_path), device=GPU, verbose=False, conf=0.15, iou=0.35)[0]
                raw_boxes = []
                for box in (result.boxes or []):
                    raw_boxes.append(
                        {
                            "box": box.xyxy[0].cpu().numpy().tolist(),
                            "cls": int(box.cls[0]),
                            "score": float(box.conf[0]),
                        }
                    )

                fixed_boxes = apply_fixes(raw_boxes)
                total_before += len(raw_boxes)
                total_after += len(fixed_boxes)

                out_img = draw_boxes(image.copy(), fixed_boxes)
                banner = f"[FIXED] {model_label}|{condition} raw:{len(raw_boxes)}->kept:{len(fixed_boxes)}"
                cv2.rectangle(out_img, (0, 0), (out_img.shape[1], 20), (0, 0, 0), -1)
                cv2.putText(out_img, banner, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
                cv2.imwrite(str(dst_dir / file_name), out_img)

            reduced_pct = 100 * (total_before - total_after) / max(total_before, 1)
            print(f"    Done: before={total_before} after={total_after} reduced={reduced_pct:.1f}%")

    print(f"\nAll done. Fixed images at: {OUT_DIR}")


if __name__ == "__main__":
    run()
