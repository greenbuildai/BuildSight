#!/usr/bin/env python3
"""
adaptive_postprocess.py
=======================
Full adaptive post-processing for BuildSight PPE detection.
Applies condition-aware filtering rules to YOLOv11 and YOLOv26 outputs.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

CONF_THRESHOLDS = {
    # Tightened S1 worker (0.42→0.45) and S4 worker (0.38→0.42) to reduce
    # clutter/material false positives in normal and crowded scenes.
    "S1_normal": {"worker": 0.45, "helmet": 0.32, "safety_vest": 0.50},
    "S2_dusty": {"worker": 0.32, "helmet": 0.18, "safety_vest": 0.30},
    "S3_low_light": {"worker": 0.32, "helmet": 0.18, "safety_vest": 0.30},
    "S4_crowded": {"worker": 0.42, "helmet": 0.24, "safety_vest": 0.36},
}

NMS_IOU = {
    "S1_normal": 0.40,
    "S2_dusty": 0.50,
    "S3_low_light": 0.50,
    "S4_crowded": 0.35,
}

CLS_NAMES = {0: "helmet", 1: "safety_vest", 2: "worker"}
CLS_COLORS = {0: (0, 255, 0), 1: (0, 165, 255), 2: (255, 100, 0)}
MAX_BOX_AREA_FRACTION = 0.20
MAX_BOX_AREA_FRACTION_BY_CONDITION = {"S4_crowded": 0.30}
WORKER_MIN_HUMAN_SCORE = {
    # Raised S1 (0.43→0.46), S2 (0.38→0.41), S4 (0.30→0.36) to reject
    # material sacks, cement bags, and squatting-object detections.
    "S1_normal": 0.46,
    "S2_dusty": 0.41,
    "S3_low_light": 0.36,
    "S4_crowded": 0.36,
}
WORKER_MIN_PIXEL_HEIGHT = {
    # Raised minimums — small blobs below these are almost always clutter.
    # S1: 24→30, S2: 22→26, S3: 20→22, S4: 18→22
    "S1_normal": 30,
    "S2_dusty": 26,
    "S3_low_light": 22,
    "S4_crowded": 22,
}
CONDITIONS = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]
GPU = 0

CONDITION_SPLIT_JSON = Path("/nfsshare/joseva/condition_eval_results/val_condition_splits.json")
VAL_IMG_DIR = Path("/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/images/val")
OUT_DIR = Path("/nfsshare/joseva/val_annotated_adaptive_v2")
LOG_DIR = Path("/nfsshare/joseva/logs")

MODEL_PATHS = {
    "yolo11": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv11/runs/detect/runs/train/buildsight/weights/best.pt",
    "yolo26": "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)/YOLOv26/runs/detect/runs/train/buildsight/weights/best.pt",
}


def iou(box_a, box_b):
    xi1 = max(box_a[0], box_b[0])
    yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2])
    yi2 = min(box_a[3], box_b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    union = (
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        - inter
    )
    return inter / max(union, 1e-6)


def is_valid_aspect(box, cls_name):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if height == 0:
        return False
    ratio = width / height

    if cls_name == "worker":
        # Tightened upper bound: 1.25→0.95. Real standing/crouching workers are
        # taller than wide. Cement bags, buckets, and scaffolding cross-sections
        # are typically wider than tall (ratio > 0.95) and are rejected here.
        return 0.15 <= ratio <= 0.95
    if cls_name == "helmet":
        return 0.40 <= ratio <= 2.0
    if cls_name == "safety_vest":
        return 0.25 <= ratio <= 1.80
    return True


def has_worker_overlap(ppe_box, worker_boxes, min_iou=0.08):
    px_c = (ppe_box[0] + ppe_box[2]) / 2
    py_c = (ppe_box[1] + ppe_box[3]) / 2
    for worker_box in worker_boxes:
        if iou(ppe_box, worker_box) >= min_iou:
            return True
        wx1, wy1, wx2, wy2 = worker_box
        w_expand = (wx2 - wx1) * 0.20
        h_expand = (wy2 - wy1) * 0.20
        if (
            wx1 - w_expand <= px_c <= wx2 + w_expand
            and wy1 - h_expand <= py_c <= wy2 + h_expand
        ):
            return True
    return False


def suppress_large_by_small(boxes, overlap_thresh=0.30):
    if not boxes:
        return []

    areas = [
        (box["box"][2] - box["box"][0]) * (box["box"][3] - box["box"][1])
        for box in boxes
    ]
    median_area = sorted(areas)[len(areas) // 2]
    small_boxes = [box for box, area in zip(boxes, areas) if area <= median_area]
    result = []

    for box, area in zip(boxes, areas):
        if area > median_area * 3:
            overlapping_small = [small for small in small_boxes if iou(box["box"], small["box"]) > overlap_thresh]
            if overlapping_small:
                if box["score"] > max(small["score"] for small in overlapping_small) + 0.15:
                    result.append(box)
                continue
        result.append(box)

    return result


def cross_class_nms(boxes, iou_thresh=0.45):
    """
    Historical name kept for compatibility.
    Cross-class suppression is intentionally disabled because worker/PPE overlap
    is expected and removing one class harms association accuracy.
    """
    boxes = sorted(boxes, key=lambda item: -item["score"])
    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(boxes)):
            if not keep[j] or boxes[i]["cls"] != boxes[j]["cls"]:
                continue
            if iou(boxes[i]["box"], boxes[j]["box"]) > iou_thresh:
                keep[j] = False
    return [box for box, is_kept in zip(boxes, keep) if is_kept]


def _clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [
        max(0, min(img_w - 1, int(round(x1)))),
        max(0, min(img_h - 1, int(round(y1)))),
        max(0, min(img_w - 1, int(round(x2)))),
        max(0, min(img_h - 1, int(round(y2)))),
    ]


def _worker_support_score(box, all_boxes):
    if not all_boxes:
        return 0.0
    support = 0.0
    for other in all_boxes:
        if other["box"] == box["box"]:
            continue
        ov = iou(box["box"], other["box"])
        if other["cls"] == 2:
            support = max(support, min(1.0, ov * 2.0))
        elif other["cls"] in (0, 1):
            px1, py1, px2, py2 = other["box"]
            wx1, wy1, wx2, wy2 = box["box"]
            cx = (px1 + px2) / 2
            cy = (py1 + py2) / 2
            if wx1 <= cx <= wx2 and wy1 <= cy <= wy2:
                support = max(support, 0.75 if other["cls"] == 0 else 0.65)
    return support


def compute_worker_human_score(box, image):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clamp_box(box, w, h)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    texture_std = float(np.std(gray)) / 64.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_score = float(np.mean(grad_mag > 18.0))

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.mean(edges > 0))
    edge_score = 1.0 - min(1.0, abs(edge_density - 0.10) / 0.10)

    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1) > 0
    row_profile = edge_mask.mean(axis=1) if edge_mask.size else np.zeros((1,), dtype=np.float32)
    col_profile = edge_mask.mean(axis=0) if edge_mask.size else np.zeros((1,), dtype=np.float32)

    top = float(np.mean(row_profile[: max(1, bh // 4)]))
    middle = float(np.mean(row_profile[bh // 4 : max(bh // 4 + 1, (3 * bh) // 4)]))
    bottom = float(np.mean(row_profile[max(0, (3 * bh) // 4) :]))
    lower_mid = float(np.mean(row_profile[max(0, bh // 2) : max(bh // 2 + 1, (7 * bh) // 8)]))
    col_mean = float(np.mean(col_profile))
    col_std = float(np.std(col_profile))

    aspect = bw / max(bh, 1.0)
    aspect_score = 1.0 - min(1.0, abs(aspect - 0.42) / 0.55)

    shape_score = 0.0
    if middle > 0:
        if top < middle * 0.95:
            shape_score += 0.45
        if bottom < middle * 1.10:
            shape_score += 0.30
        if col_std > col_mean * 0.35:
            shape_score += 0.25

    # Low-contrast clothing can still form a human silhouette even if texture is weak.
    silhouette_bonus = 0.0
    if aspect <= 0.70 and bh >= 40:
        if 0.03 <= edge_density <= 0.18:
            silhouette_bonus += 0.18
        if grad_score >= 0.08:
            silhouette_bonus += 0.12

    # Material sacks/buckets tend to be squat, bottom-heavy, horizontally uniform,
    # and weak in vertical edge structure. Penalties are cumulative.
    bag_penalty = 0.0
    if aspect >= 0.60 and bh <= 95:
        if bottom > max(top, middle) * 1.10:
            bag_penalty += 0.14
        if lower_mid > middle * 1.05:
            bag_penalty += 0.10
        if col_std < max(0.02, col_mean * 0.18):
            bag_penalty += 0.12
        if grad_score < 0.07 and edge_density < 0.06:
            bag_penalty += 0.14
        if top < 0.015 and middle < 0.04:
            bag_penalty += 0.10
    if aspect >= 0.80:
        bag_penalty += 0.10
    # Very uniform color in the crop → likely flat material surface, not clothing
    if texture_std < 0.08 and grad_score < 0.10:
        bag_penalty += 0.12

    score = (
        0.34 * np.clip(aspect_score, 0.0, 1.0)
        + 0.14 * np.clip(texture_std, 0.0, 1.0)
        + 0.18 * np.clip(edge_score, 0.0, 1.0)
        + 0.22 * np.clip(shape_score, 0.0, 1.0)
        + 0.12 * np.clip(grad_score, 0.0, 1.0)
    )
    score += silhouette_bonus
    score -= bag_penalty
    return float(np.clip(score, 0.0, 1.0))


def _is_blue_bucket(crop_bgr) -> float:
    """
    Returns a blue-bucket probability [0, 1] based on HSV blue-hue dominance.
    Blue buckets on Indian construction sites typically cluster in H=100-130 (HSV),
    high saturation (>90), mid-high value. Returns > 0.5 when blue dominates.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    blue_lo = np.array([95, 80, 60], dtype=np.uint8)
    blue_hi = np.array([135, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lo, blue_hi)
    blue_frac = float(np.mean(blue_mask > 0))
    return min(1.0, blue_frac * 2.5)


def _is_cement_bag(crop_bgr) -> float:
    """
    Returns a cement/sand-bag probability [0, 1].
    Cement bags: gray/white/tan tones, low saturation, horizontally wide,
    uniform texture (low gradient variance). Score > 0.55 → likely bag.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    val = float(np.mean(hsv[:, :, 2]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_std = float(np.std(grad_mag))

    score = 0.0
    # Low saturation = not vivid = plausible bag/material
    if sat < 55:
        score += 0.30
    elif sat < 80:
        score += 0.15
    # Typical bag brightness range
    if 60 < val < 200:
        score += 0.15
    # Uniform surface = low gradient variation
    if grad_std < 18:
        score += 0.30
    elif grad_std < 28:
        score += 0.15
    # Wide horizontal blob = bag shape
    if w > 0 and h > 0 and (w / h) > 0.85:
        score += 0.15
    return min(1.0, score)


def _is_scaffolding(crop_bgr) -> float:
    """
    Returns a scaffolding probability [0, 1].
    Scaffolding: strong regular vertical/horizontal line structure, metallic gray,
    high edge density in a grid-like pattern. Score > 0.55 → likely scaffolding.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Check for strong horizontal/vertical line dominance via Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                             minLineLength=12, maxLineGap=4)
    line_count = len(lines) if lines is not None else 0

    score = 0.0
    # Very high edge density in a structured pattern = scaffolding
    if edge_density > 0.20:
        score += 0.35
    elif edge_density > 0.14:
        score += 0.20
    # Many short structural lines = scaffolding grid
    h, w = crop_bgr.shape[:2]
    crop_area = max(h * w, 1)
    line_density = line_count / (crop_area / 400)  # lines per 400px²
    if line_density > 3.0:
        score += 0.35
    elif line_density > 1.5:
        score += 0.20

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:, :, 1]))
    # Low saturation metallic color
    if sat < 40:
        score += 0.15
    return min(1.0, score)


def suppress_hard_negatives(boxes, image, condition):
    """
    Hard negative suppression for known Indian construction-site clutter:
      - Blue buckets / barrels  (strong blue HSV dominance)
      - Cement / sand bags      (gray/white, wide, uniform texture)
      - Scaffolding             (high-edge metallic grid structure)

    Only applied to worker (cls=2) boxes. PPE boxes are not suppressed here
    because the PPE-to-worker anchor check handles floating detections.

    Returns (filtered_boxes, suppressed_count).
    """
    if image is None:
        return boxes, 0

    h_img, w_img = image.shape[:2]
    kept = []
    suppressed = 0

    # Thresholds — slightly looser in degraded conditions to avoid recall loss
    blue_thresh = 0.55 if condition == "S1_normal" else 0.65
    bag_thresh = 0.62 if condition == "S1_normal" else 0.70
    scaffold_thresh = 0.60 if condition == "S1_normal" else 0.68

    for box in boxes:
        if box["cls"] != 2:
            kept.append(box)
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in box["box"]]
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y2 = max(y1 + 1, min(y2, h_img))
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            kept.append(box)
            continue

        bucket_score = _is_blue_bucket(crop)
        bag_score = _is_cement_bag(crop)
        scaffold_score = _is_scaffolding(crop)

        # High-confidence individual classifier suppresses the detection
        if bucket_score >= blue_thresh:
            suppressed += 1
            continue
        if bag_score >= bag_thresh:
            suppressed += 1
            continue
        if scaffold_score >= scaffold_thresh:
            suppressed += 1
            continue

        # Combined — two medium-confidence signals together are enough to suppress
        combined = max(
            0.5 * bucket_score + 0.5 * bag_score,
            0.5 * bucket_score + 0.5 * scaffold_score,
            0.5 * bag_score + 0.5 * scaffold_score,
        )
        if combined >= 0.58 and box["score"] < 0.75:
            suppressed += 1
            continue

        kept.append(box)

    return kept, suppressed


def suppress_material_workers(boxes, image, condition):
    if image is None:
        return boxes, 0

    kept = []
    rejected = 0
    min_score = WORKER_MIN_HUMAN_SCORE.get(condition, 0.50)
    min_height = WORKER_MIN_PIXEL_HEIGHT.get(condition, 20)

    for box in boxes:
        if box["cls"] != 2:
            kept.append(box)
            continue

        x1, y1, x2, y2 = box["box"]
        height = y2 - y1
        if height < min_height:
            kept.append(box)
            continue

        human_score = compute_worker_human_score(box["box"], image)
        support_score = _worker_support_score(box, boxes)
        combined = max(human_score, 0.65 * human_score + 0.35 * support_score)
        box["human_score"] = round(combined, 3)
        aspect = (x2 - x1) / max((y2 - y1), 1.0)
        short_wide_material = height <= 112 and aspect >= 0.60 and support_score < 0.45
        weak_human_shape = aspect >= 0.56 and height <= 120 and human_score < (min_score - 0.06)

        slender_worker = aspect <= 0.72 and (y2 - y1) >= (min_height + 8)
        if short_wide_material and combined < (min_score + 0.08):
            rejected += 1
            continue
        if weak_human_shape and combined < min_score and box["score"] < 0.78 and not slender_worker:
            rejected += 1
            continue
        if combined < min_score and support_score < 0.60 and box["score"] < 0.85 and not (slender_worker and combined >= (min_score - 0.08)):
            rejected += 1
            continue
        kept.append(box)

    return kept, rejected


def vertical_position_filter(boxes, img_h, condition):
    if condition != "S1_normal":
        return boxes

    filtered = []
    for box in boxes:
        if box["cls"] == 2:
            x1, y1, x2, y2 = box["box"]
            cy = (y1 + y2) / 2
            box_h = y2 - y1
            cy_frac = cy / img_h
            box_h_frac = box_h / img_h
            if cy_frac < 0.45 and box_h_frac > 0.08 and box["score"] < 0.90:
                continue
        filtered.append(box)
    return filtered


def area_filtered_boxes(boxes, img_area, area_fraction):
    kept = []
    worker_boxes = [box for box in boxes if CLS_NAMES[box["cls"]] == "worker"]
    for box in boxes:
        x1, y1, x2, y2 = box["box"]
        area = (x2 - x1) * (y2 - y1)
        if area / img_area <= area_fraction:
            kept.append(box)
            continue
        if CLS_NAMES[box["cls"]] == "worker":
            overlapping_workers = [
                worker
                for worker in worker_boxes
                if worker is not box and iou(box["box"], worker["box"]) > 0.30
            ]
            if overlapping_workers:
                kept.append(box)
    return kept


def apply_all_rules(raw_boxes, condition, img_w, img_h, image=None, track_context=None):
    img_area = img_w * img_h
    cond_conf = CONF_THRESHOLDS[condition]
    area_fraction = MAX_BOX_AREA_FRACTION_BY_CONDITION.get(condition, MAX_BOX_AREA_FRACTION)
    stats = {"raw": len(raw_boxes)}

    boxes = [box for box in raw_boxes if box["score"] >= cond_conf[CLS_NAMES[box["cls"]]]]
    stats["after_conf"] = len(boxes)

    boxes = area_filtered_boxes(boxes, img_area, area_fraction)
    stats["after_area"] = len(boxes)

    boxes = [box for box in boxes if is_valid_aspect(box["box"], CLS_NAMES[box["cls"]])]
    stats["after_aspect"] = len(boxes)

    boxes = suppress_large_by_small(boxes)
    stats["after_large_suppress"] = len(boxes)

    boxes = cross_class_nms(boxes, NMS_IOU[condition])
    stats["after_cross_nms"] = len(boxes)

    boxes = vertical_position_filter(boxes, img_h, condition)
    stats["after_vertical"] = len(boxes)

    boxes, suppressed_material = suppress_material_workers(boxes, image, condition)
    stats["after_worker_validation"] = len(boxes)
    stats["suppressed_material_workers"] = suppressed_material

    # Hard negative suppression: blue buckets, cement bags, scaffolding
    boxes, suppressed_hard = suppress_hard_negatives(boxes, image, condition)
    stats["after_hard_negative"] = len(boxes)
    stats["suppressed_hard_negatives"] = suppressed_hard

    worker_boxes = [box["box"] for box in boxes if box["cls"] == 2]
    if worker_boxes:
        anchored = []
        for box in boxes:
            if box["cls"] in (0, 1) and not has_worker_overlap(box["box"], worker_boxes):
                continue
            anchored.append(box)
        boxes = anchored
    else:
        ppe_boxes = [box for box in boxes if box["cls"] in (0, 1)]
        worker_free = [box for box in boxes if box["cls"] not in (0, 1)]
        ppe_boxes = sorted(ppe_boxes, key=lambda item: -item["score"])[:3]
        boxes = worker_free + ppe_boxes

    stats["after_ppe_anchor"] = len(boxes)
    stats["final"] = len(boxes)
    return boxes, stats


def draw_boxes(image, boxes, condition, model_label, stats):
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box["box"]]
        color = CLS_COLORS[box["cls"]]
        label = f"{CLS_NAMES[box['cls']]} {box['score']:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(image, (x1, y1 - text_h - 4), (x1 + text_w + 2, y1), color, -1)
        cv2.putText(image, label, (x1 + 1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    banner = (
        f"[ADAPTIVE] {model_label}|{condition} "
        f"raw:{stats['raw']}->final:{stats['final']} "
        f"(-{stats['raw'] - stats['final']})"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(image, banner, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
    return image


def run():
    from ultralytics import YOLO

    splits = json.loads(CONDITION_SPLIT_JSON.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []

    for model_key, model_path in MODEL_PATHS.items():
        model_label = "YOLOv11" if model_key == "yolo11" else "YOLOv26"
        print(f"\n=== {model_label} ===")
        model = YOLO(model_path)

        for condition in CONDITIONS:
            file_names = splits.get(condition, [])
            if not file_names:
                continue

            dst_dir = OUT_DIR / condition / model_label
            dst_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {condition}: {len(file_names)} images")

            for file_name in file_names:
                img_path = VAL_IMG_DIR / file_name
                if not img_path.exists():
                    continue

                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                height, width = image.shape[:2]

                result = model.predict(str(img_path), device=GPU, verbose=False, conf=0.07, iou=0.35)[0]
                raw_boxes = []
                for box in (result.boxes or []):
                    raw_boxes.append(
                        {
                            "box": box.xyxy[0].cpu().numpy().tolist(),
                            "cls": int(box.cls[0]),
                            "score": float(box.conf[0]),
                        }
                    )

                final_boxes, stats = apply_all_rules(raw_boxes, condition, width, height)
                out_img = draw_boxes(image.copy(), final_boxes, condition, model_label, stats)
                cv2.imwrite(str(dst_dir / file_name), out_img)

                csv_rows.append(
                    {
                        "model": model_label,
                        "condition": condition,
                        "file": file_name,
                        "raw": stats["raw"],
                        "after_conf": stats["after_conf"],
                        "after_area": stats["after_area"],
                        "after_aspect": stats["after_aspect"],
                        "after_large_suppress": stats["after_large_suppress"],
                        "after_cross_nms": stats["after_cross_nms"],
                        "after_vertical": stats["after_vertical"],
                        "after_ppe_anchor": stats["after_ppe_anchor"],
                        "final": stats["final"],
                    }
                )

    csv_path = LOG_DIR / "adaptive_postprocess_v2_summary.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nDone. Output: {OUT_DIR}")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    run()
