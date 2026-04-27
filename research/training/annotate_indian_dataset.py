"""
annotate_indian_dataset.py
==========================
BuildSight AI — PPE Detection Annotation Pipeline (FIXED)
Green Build AI | IGBC AP | NBC 2016 Aligned

FINAL 3-CLASS SCHEMA (sequential IDs, no gaps):
  Class ID 0 → helmet        (hard_hat merged in)
  Class ID 1 → safety_vest   (high_vis_jacket merged in)
  Class ID 2 → worker

BUGS FIXED IN THIS VERSION:
  BUG-01 / FIX-01 — Two-pass detection: worker boxes first, then PPE per-crop
  BUG-02 / FIX-02 — Per-class NMS only; cross-class suppression eliminated
  BUG-03 / FIX-03 — Per-class confidence thresholds loaded from config
  BUG-04 / FIX-03+04 — Low-light CLAHE + gamma preprocessing
  BUG-05 / FIX-01 — Per-worker PPE sub-detection for crowded scenes
  BUG-06 / FIX-03+04 — Per-condition threshold reduction for consistency

Output:
  Dataset/Final_Annotated_Dataset/
    images/train|val|test/
    labels/train|val|test/        ← YOLO bbox .txt (class IDs 0-3)
    labels_seg/train|val|test/    ← YOLO polygon .txt (class IDs 0-3)
    annotations/
      instances_train.json        ← COCO JSON (category IDs 0-3)
      instances_val.json
      instances_test.json
    data.yaml                     ← 4-class config

Usage:
  python annotate_indian_dataset.py [--test-batch N] [--skip-sam] [--conditions X,Y]
"""

import os
import sys
import re
import cv2

# Ensure we can find pipeline_config.py if run from root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import math
import shutil
import hashlib
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from typing import Optional

# Ensure we can find pipeline_config.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import (
    BASE_DIR, DATA_DIR, OUTPUT_DIR,
    DINO_WEIGHTS, SAM_WEIGHTS,
    CLASS_ID, CLASS_NAMES, NUM_CLASSES,
    WORKER_TEXT_PROMPT, PPE_TEXT_PROMPT, PPE_CROP_TEXT_PROMPT,
    DINO_BOX_THRESHOLD, DINO_TEXT_THRESHOLD,
    DINO_CROP_BOX_THRESHOLD, DINO_CROP_TEXT_THRESHOLD,
    PHRASE_TO_CLASS,
    get_threshold,
    WORKER_BOX_MAX_FRACTION, WORKER_CROP_PADDING,
    NMS_IOU_THRESHOLD,
    CLAHE_PARAMS, GAMMA_CORRECTION,
    LOW_LIGHT_BRIGHTNESS_THRESHOLD,
    DUSTY_STD_THRESHOLD, DUSTY_SATURATION_THRESHOLD,
    CROWDED_WORKER_COUNT_THRESHOLD, CROWDED_QUICK_DETECT_THRESHOLD,
    WORKER_MIN_HEIGHT_FOR_PPE, WORKER_CROP_UPSCALE_FACTOR,
    PPE_BOX_MIN_PX, WORKER_BOX_MIN_PX, WORKER_MIN_COLOR_STD,
    PPE_CONTAINMENT_TOL_H, PPE_CONTAINMENT_TOL_V_UP, PPE_CONTAINMENT_TOL_V_DOWN,
    HELMET_MIN_ASPECT_RATIO, HELMET_MAX_ASPECT_RATIO, HELMET_MAX_CENTRE_Y_FRACTION,
    VEST_BOX_MAX_HEIGHT_FRACTION, VEST_WORKER_MIN_H_FOR_CHECK,
    CROWDED_SCENE_THRESHOLD,
    LOW_CONF_THRESHOLD,
    SEQUENCE_GAP_MS, SPLIT_RATIOS,
    EXPORT_YOLO_BBOX, EXPORT_YOLO_SEG, EXPORT_COCO_JSON,
    FOLDER_TO_CONDITION, CONDITIONS,
)

import groundingdino as _gd
_GD_PKG_DIR = Path(_gd.__file__).parent
DINO_CONFIG  = _GD_PKG_DIR / "config" / "GroundingDINO_SwinT_OGC.py"

# Synthetic image detection patterns
SYNTHETIC_PATTERNS = [
    re.compile(r"^advanced_", re.IGNORECASE),
    re.compile(r"^basic_dust", re.IGNORECASE),
    re.compile(r"^original_style", re.IGNORECASE),
    re.compile(r"^digital_", re.IGNORECASE),
    re.compile(r"^\d{13,}"),
]

# ─────────────────────────────────────────────────────────────────────────────
# COCO CATEGORY DEFINITIONS (3-class, sequential IDs)
# ─────────────────────────────────────────────────────────────────────────────

COCO_CATEGORIES = [
    {"supercategory": "ppe",    "id": CLASS_ID["helmet"],        "name": "helmet"},
    {"supercategory": "ppe",    "id": CLASS_ID["safety_vest"],   "name": "safety_vest"},
    {"supercategory": "person", "id": CLASS_ID["worker"],        "name": "worker"},
]


# ─────────────────────────────────────────────────────────────────────────────
# FIX-05: CONDITION AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def count_persons_quick(image_bgr: np.ndarray, dino_model, device: str) -> int:
    """
    Run a fast worker detection pass to count persons.
    Used by detect_scene_condition to check for crowded scenes.
    """
    from groundingdino.util.inference import load_image, predict
    h, w = image_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    # Use PIL image instead of file path for in-memory inference
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(pil_img)

    try:
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image_tensor,
            caption=WORKER_TEXT_PROMPT,
            box_threshold=CROWDED_QUICK_DETECT_THRESHOLD,
            text_threshold=0.15,
            device=device,
        )
        count = sum(1 for p in phrases if phrase_to_class_id(p) == CLASS_ID["worker"])
        return count
    except Exception:
        return 0


def detect_scene_condition(image_bgr: np.ndarray, dino_model, device: str,
                           folder_condition: str = None) -> str:
    """
    FIX-05: Auto-classify a frame into one of 4 site conditions.
    Returns: 'normal' | 'dusty' | 'low_light' | 'crowded'

    folder_condition (optional): canonical condition from folder name —
    used as a tiebreaker when image heuristics are ambiguous.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_brightness  = float(np.std(gray))

    # Low-light detection: mean brightness below threshold
    # FIX-04: CRITICAL — dark interior rebar frame only gets 1 detection
    # because brightness-based pre-screening was missing. Now detected first.
    if mean_brightness < LOW_LIGHT_BRIGHTNESS_THRESHOLD:
        return 'low_light'

    # Dusty detection: low contrast (low std) + desaturated color cast
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    if std_brightness < DUSTY_STD_THRESHOLD and mean_saturation < DUSTY_SATURATION_THRESHOLD:
        return 'dusty'

    # Crowded detection: quick lightweight person count
    person_count = count_persons_quick(image_bgr, dino_model, device)
    if person_count >= CROWDED_WORKER_COUNT_THRESHOLD:
        return 'crowded'

    # Fall back to folder-derived condition if available
    if folder_condition in ('dusty', 'low_light', 'crowded'):
        return folder_condition

    return 'normal'


# ─────────────────────────────────────────────────────────────────────────────
# FIX-04: PER-CONDITION PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def apply_gamma_correction(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction. gamma > 1 brightens, gamma < 1 darkens."""
    if gamma == 1.0:
        return image_bgr
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8
    )
    return cv2.LUT(image_bgr, table)


def preprocess_for_inference(image_bgr: np.ndarray, condition: str) -> np.ndarray:
    """
    FIX-04: Apply condition-specific preprocessing.
    Returns an ENHANCED COPY for inference — source image is NEVER modified.

    Conditions:
      normal/crowded → no change
      dusty          → CLAHE (clipLimit=3.0, tileGrid=8×8)
      low_light      → CLAHE (clipLimit=4.0, tileGrid=8×8) + gamma=1.8
    """
    img = image_bgr.copy()

    clahe_params = CLAHE_PARAMS.get(condition)
    gamma        = GAMMA_CORRECTION.get(condition, 1.0)

    # CLAHE: applied to L-channel of LAB colorspace to preserve color balance
    if clahe_params is not None:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clahe_params["clip_limit"],
            tileGridSize=clahe_params["tile_grid_size"],
        )
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma correction
    if gamma != 1.0:
        img = apply_gamma_correction(img, gamma)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# FIX-02: PER-CLASS NMS (cross-class suppression ELIMINATED)
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute IoU between two [x1,y1,x2,y2] absolute boxes.
    Used by per-class NMS.
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def per_class_nms(detections: list) -> list:
    """
    FIX-02: Apply NMS independently per class.
    A helmet box at IoU=0.85 with a worker box is NEVER suppressed.
    Only two boxes of the SAME class in the same spatial location are suppressed.

    detections: list of dicts with keys: class_id, score, xyxy (list [x1,y1,x2,y2])
    Returns: filtered list.
    """
    # FIX-02: Group by class_id — NEVER apply NMS across different classes
    by_class = defaultdict(list)
    for det in detections:
        by_class[det["class_id"]].append(det)

    kept = []
    for class_id, class_dets in by_class.items():
        class_name = CLASS_NAMES[class_id]
        iou_thresh = NMS_IOU_THRESHOLD.get(class_name, 0.45)

        # Greedy NMS: sort by score descending, suppress lower-score duplicates
        class_dets.sort(key=lambda x: x["score"], reverse=True)
        remaining = list(class_dets)
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            # Only suppress boxes of the SAME class that overlap too much
            remaining = [
                d for d in remaining
                if compute_iou(best["xyxy"], d["xyxy"]) < iou_thresh
            ]

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_synthetic(filename: str) -> bool:
    for pattern in SYNTHETIC_PATTERNS:
        if pattern.match(filename):
            return True
    return False


def extract_timestamp_ms(filename: str):
    stem = Path(filename).stem
    match = re.match(r"^(\d{13,})$", stem)
    return int(match.group(1)) if match else None


def cluster_into_sequences(filenames: list, gap_ms: int = SEQUENCE_GAP_MS) -> dict:
    """Group consecutive timestamp-named images into SAMURAI sequences."""
    sorted_files = sorted(filenames)
    assignments  = {}
    seq_idx      = 0
    current_seq  = []

    for fname in sorted_files:
        ts = extract_timestamp_ms(fname)
        if ts is None:
            assignments[fname] = (seq_idx, 0)
            seq_idx += 1
            continue

        if not current_seq:
            current_seq = [(fname, ts)]
        else:
            prev_ts = current_seq[-1][1]
            if abs(ts - prev_ts) <= gap_ms:
                current_seq.append((fname, ts))
            else:
                for frame_idx, (f, _) in enumerate(current_seq):
                    assignments[f] = (seq_idx, frame_idx)
                seq_idx += 1
                current_seq = [(fname, ts)]

    for frame_idx, (f, _) in enumerate(current_seq):
        assignments[f] = (seq_idx, frame_idx)

    return assignments


def phrase_to_class_id(phrase: str):
    """Map a GroundingDINO phrase to a class_id. Returns None if no match."""
    phrase_lower = phrase.lower()
    for key, cls_id in PHRASE_TO_CLASS.items():
        if key in phrase_lower:
            return cls_id
    return None


def image_tensor_from_array(image_bgr: np.ndarray):
    """Convert a BGR numpy array to a GroundingDINO-compatible image tensor."""
    import torchvision.transforms as T
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    transform = T.Compose([
        T.Resize((800, 1333)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return np.array(rgb), transform(pil)


def dino_predict_on_array(dino_model, image_bgr: np.ndarray,
                          text_prompt: str, device: str,
                          box_threshold: Optional[float] = None,
                          text_threshold: Optional[float] = None):
    """
    Run GroundingDINO prediction on a BGR numpy array (not a file path).
    Returns (xyxy_absolute, logits, phrases) for the given image.

    box_threshold / text_threshold override the global defaults when supplied.
    Pass DINO_CROP_BOX_THRESHOLD for crop-level calls (Fix-UP Step 1).
    """
    from groundingdino.util.inference import predict

    if box_threshold is None:
        box_threshold = DINO_BOX_THRESHOLD
    if text_threshold is None:
        text_threshold = DINO_TEXT_THRESHOLD

    image_source, image_tensor = image_tensor_from_array(image_bgr)
    h, w = image_bgr.shape[:2]

    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    # Convert normalized [cx,cy,w,h] → absolute [x1,y1,x2,y2]
    if len(boxes) > 0:
        xyxy = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
        xyxy[:, :2] -= xyxy[:, 2:] / 2
        xyxy[:, 2:] += xyxy[:, :2]
    else:
        xyxy = torch.zeros((0, 4))

    return xyxy.cpu().numpy(), logits.cpu().numpy(), phrases


def boxes_to_coco_format(xyxy_boxes: np.ndarray, img_w: int, img_h: int) -> list:
    """Convert [x1,y1,x2,y2] absolute boxes to COCO [x,y,w,h]."""
    result = []
    for box in xyxy_boxes:
        x1 = max(0.0, float(box[0]))
        y1 = max(0.0, float(box[1]))
        x2 = min(float(img_w), float(box[2]))
        y2 = min(float(img_h), float(box[3]))
        result.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    return result


def coco_box_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """COCO [x,y,w,h] → YOLO normalized [cx,cy,w,h]."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def polygon_to_yolo_seg(polygon: list, img_w: int, img_h: int) -> str:
    """Flatten and normalize a polygon [[x,y,...]] for YOLO seg format."""
    coords = polygon[0] if polygon else []
    parts  = []
    for i in range(0, len(coords) - 1, 2):
        parts.append(f"{coords[i] / img_w:.6f}")
        parts.append(f"{coords[i+1] / img_h:.6f}")
    return " ".join(parts)


def split_files(filenames: list, ratios: dict = SPLIT_RATIOS, seed: int = 42) -> dict:
    """Stratified random split into train/val/test."""
    files = sorted(filenames)
    rng   = random.Random(seed)
    rng.shuffle(files)
    n       = len(files)
    n_train = math.floor(n * ratios["train"])
    n_val   = math.floor(n * ratios["val"])
    return {
        "train": files[:n_train],
        "val":   files[n_train : n_train + n_val],
        "test":  files[n_train + n_val :],
    }


def auto_instance_flags(score: float, condition: str, n_workers: int,
                        bbox: list, img_w: int, img_h: int) -> dict:
    """Derive per-instance attribute flags from detection context."""
    flags = {
        "dust_occluded":     condition == "dusty"     and score < 0.50,
        "low_visibility":    condition == "low_light" and score < 0.50,
        "low_conf":          score < LOW_CONF_THRESHOLD,
        "truncated":         False,
        "crowd_occluded":    condition == "crowded"   and n_workers >= CROWDED_SCENE_THRESHOLD,
        "overexposed_glare": False,
        "no_ppe":            False,
        "no_ppe_visible":    False,
    }
    x, y, w, h = bbox
    if x <= 5 or y <= 5 or (x + w) >= (img_w - 5) or (y + h) >= (img_h - 5):
        flags["truncated"] = True
    return flags


def make_dust_zone_placeholder(img_w: int, img_h: int) -> list:
    return [[0, 0, img_w, 0, img_w, img_h, 0, img_h]]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_models(skip_sam: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  GPU Turbo: cudnn.benchmark=True, TF32 enabled")

    print("Loading GroundingDINO...")
    from groundingdino.util.inference import load_model
    dino_model = load_model(str(DINO_CONFIG), str(DINO_WEIGHTS), device=device)

    sam_predictor = None
    if not skip_sam:
        print("Loading SAM vit_b...")
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM_WEIGHTS)).to(device=device)
        if device == "cuda":
            sam = sam.half()   # FP16 — ~2x faster SAM inference on RTX GPU
        sam_predictor = SamPredictor(sam)

    return dino_model, sam_predictor, device


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────

def setup_output_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images"     / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels"     / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels_seg" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "annotations").mkdir(parents=True, exist_ok=True)
    print(f"Output directories: {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# FIX-01: TWO-PASS DETECTION WITH PER-WORKER PPE SUB-DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def run_worker_pass(inf_image: np.ndarray, dino_model, device: str,
                    condition: str) -> list:
    """
    FIX-01 / PASS 1: Detect worker (person) boxes on the full image.
    Returns list of dicts: {class_id, score, xyxy, class_name}
    """
    h, w = inf_image.shape[:2]
    worker_threshold = get_threshold("worker", condition)   # FIX-03

    xyxy_arr, logits, phrases = dino_predict_on_array(
        dino_model, inf_image, WORKER_TEXT_PROMPT, device
    )

    detections = []
    for i in range(len(xyxy_arr)):
        cls_id = phrase_to_class_id(phrases[i])
        if cls_id != CLASS_ID["worker"]:
            continue
        score = float(logits[i])

        # FIX-03: Apply per-class confidence threshold (not a global value)
        if score < worker_threshold:
            continue

        x1, y1, x2, y2 = xyxy_arr[i]
        bw = x2 - x1
        bh = y2 - y1

        # FIX-04: Size Filtering (Anti-Machinery)
        # Prevent large construction machines from being labeled as workers.
        if bw > (w * WORKER_BOX_MAX_FRACTION) or bh > (h * WORKER_BOX_MAX_FRACTION):
            continue

        if bw < WORKER_BOX_MIN_PX or bh < WORKER_BOX_MIN_PX:
            continue

        # Anti-pictogram: safety sign human figures have flat uniform colours.
        # Real workers have complex texture — clothing patterns, skin, shadows.
        crop = inf_image[int(y1):int(y2), int(x1):int(x2)]
        if crop.size > 0 and float(crop.std()) < WORKER_MIN_COLOR_STD:
            continue

        detections.append({
            "class_id":   CLASS_ID["worker"],
            "class_name": "worker",
            "score":      score,
            "xyxy":       [float(x1), float(y1), float(x2), float(y2)],
        })

    # FIX-02: Per-class NMS on worker detections only
    return per_class_nms(detections)


def run_ppe_pass_on_worker_crop(
        inf_image: np.ndarray, worker_det: dict,
        full_img_w: int, full_img_h: int,
        dino_model, device: str, condition: str) -> list:
    """
    FIX-01 / PASS 2: Detect PPE items within a single worker's bounding box crop.
    Remaps crop-relative coordinates back to full-image space.

    Returns list of PPE detection dicts (class_id 0/1/2 only — not worker).
    """
    x1, y1, x2, y2 = worker_det["xyxy"]
    bw = x2 - x1
    bh = y2 - y1

    # FIX-01: Skip tiny workers — too small for reliable PPE sub-detection
    if bh < WORKER_MIN_HEIGHT_FOR_PPE:
        return []

    # Add padding so helmet/boots at box edges are not clipped
    pad_x = int(bw * WORKER_CROP_PADDING)
    pad_y = int(bh * WORKER_CROP_PADDING)
    cx1 = max(0, int(x1) - pad_x)
    cy1 = max(0, int(y1) - pad_y)
    cx2 = min(full_img_w, int(x2) + pad_x)
    cy2 = min(full_img_h, int(y2) + pad_y)

    crop = inf_image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return []

    crop_h, crop_w = crop.shape[:2]

    # Fix-UP Step 2: Upscale crop — more ViT patches per PPE item, boosts confidence
    if WORKER_CROP_UPSCALE_FACTOR > 1.0:
        new_w = int(crop_w * WORKER_CROP_UPSCALE_FACTOR)
        new_h = int(crop_h * WORKER_CROP_UPSCALE_FACTOR)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Fix-UP Step 1+3: aggressive crop thresholds + simplified prompt
    xyxy_arr, logits, phrases = dino_predict_on_array(
        dino_model, crop, PPE_CROP_TEXT_PROMPT, device,
        box_threshold=DINO_CROP_BOX_THRESHOLD,
        text_threshold=DINO_CROP_TEXT_THRESHOLD,
    )

    ppe_dets = []
    for i in range(len(xyxy_arr)):
        cls_id = phrase_to_class_id(phrases[i])
        if cls_id is None or cls_id == CLASS_ID["worker"]:
            continue   # this pass is PPE-only

        class_name = CLASS_NAMES[cls_id]
        score = float(logits[i])

        # FIX-03: Per-class confidence threshold per condition
        if score < get_threshold(class_name, condition):
            continue

        # Map crop-relative coordinates back to full image space.
        # If crop was upscaled, divide by upscale factor first to get
        # original crop-space coordinates before adding the crop offset.
        bx1, by1, bx2, by2 = xyxy_arr[i]
        scale = WORKER_CROP_UPSCALE_FACTOR if WORKER_CROP_UPSCALE_FACTOR > 1.0 else 1.0
        abs_x1 = float(bx1) / scale + cx1
        abs_y1 = float(by1) / scale + cy1
        abs_x2 = float(bx2) / scale + cx1
        abs_y2 = float(by2) / scale + cy1

        # Clip to full image bounds
        abs_x1 = max(0.0, min(float(full_img_w), abs_x1))
        abs_y1 = max(0.0, min(float(full_img_h), abs_y1))
        abs_x2 = max(0.0, min(float(full_img_w), abs_x2))
        abs_y2 = max(0.0, min(float(full_img_h), abs_y2))

        # PPE Containment Check: discard detections whose centre lands outside the
        # original (unpadded) worker box. Prevents vest/helmet false positives on
        # adjacent safety signs that fall into the crop's padding region.
        ppe_cx = (abs_x1 + abs_x2) / 2.0
        ppe_cy = (abs_y1 + abs_y2) / 2.0
        tol_x  = (x2 - x1) * PPE_CONTAINMENT_TOL_H
        tol_up = (y2 - y1) * PPE_CONTAINMENT_TOL_V_UP
        tol_dn = (y2 - y1) * PPE_CONTAINMENT_TOL_V_DOWN
        if not (x1 - tol_x <= ppe_cx <= x2 + tol_x and
                y1 - tol_up <= ppe_cy <= y2 + tol_dn):
            continue

        box_w = abs_x2 - abs_x1
        box_h = abs_y2 - abs_y1
        if box_w < PPE_BOX_MIN_PX or box_h < PPE_BOX_MIN_PX:
            continue

        # PERMANENT HALLUCINATION FILTER:
        # A helmet physically cannot be larger than 40% of the worker's total height.
        # This violently deletes any DINO hallucination that tries to classify a chest vest as a helmet.
        if class_name == "helmet" and box_h > (bh * 0.40):
            continue

        # Guard 1 — Helmet aspect ratio filter
        # Real helmets are roughly square (width/height 0.5–1.8).
        # Shirt collars / wide upper-torso regions fall outside this range.
        if class_name == "helmet":
            aspect = (box_w / box_h) if box_h > 0 else 0
            if not (HELMET_MIN_ASPECT_RATIO <= aspect <= HELMET_MAX_ASPECT_RATIO):
                continue

        # Guard 1b — Helmet vertical position filter
        # A helmet must sit in the TOP of the worker box. Shirt/chest detections
        # land in the middle or lower portion and are rejected here.
        if class_name == "helmet":
            worker_h = y2 - y1
            max_centre_y = y1 + HELMET_MAX_CENTRE_Y_FRACTION * worker_h
            if ppe_cy > max_centre_y:
                continue

        # Guard 2 — Vest height fraction filter (close-up workers only)
        # Only applied when the worker box is large enough (>= VEST_WORKER_MIN_H_FOR_CHECK px)
        # to reliably distinguish torso from full-body. Distant small worker boxes skip
        # this check because their vest legitimately spans 80–90% of the tight box.
        if class_name == "safety_vest":
            worker_h = y2 - y1
            if worker_h >= VEST_WORKER_MIN_H_FOR_CHECK and (box_h / worker_h) > VEST_BOX_MAX_HEIGHT_FRACTION:
                continue

        ppe_dets.append({
            "class_id":   cls_id,
            "class_name": class_name,
            "score":      score,
            "xyxy":       [abs_x1, abs_y1, abs_x2, abs_y2],
        })

    return ppe_dets


def run_ppe_pass_on_full_image(
        inf_image: np.ndarray, worker_dets: list,
        dino_model, device: str, condition: str) -> list:
    """
    B2: Full-image PPE detection pass for crowded scenes.

    Runs PPE detection on the complete image (not per-worker crops), then
    keeps only detections whose center falls within a padded worker box.
    This captures PPE on tightly-packed or partially-occluded workers where
    individual crops are too small for reliable sub-detection.

    Returns list of PPE detection dicts with full-image coordinates.
    """
    h, w = inf_image.shape[:2]

    xyxy_arr, logits, phrases = dino_predict_on_array(
        dino_model, inf_image, PPE_TEXT_PROMPT, device
    )

    # Build padded worker regions for association filtering
    # Each PPE box center must fall inside at least one expanded worker box
    padded_workers = []
    for wd in worker_dets:
        wx1, wy1, wx2, wy2 = wd["xyxy"]
        bw = wx2 - wx1
        bh = wy2 - wy1
        pad_x = bw * WORKER_CROP_PADDING
        pad_y = bh * WORKER_CROP_PADDING
        padded_workers.append((
            max(0.0, wx1 - pad_x),
            max(0.0, wy1 - pad_y),
            min(float(w), wx2 + pad_x),
            min(float(h), wy2 + pad_y),
        ))

    ppe_dets = []
    for i in range(len(xyxy_arr)):
        cls_id = phrase_to_class_id(phrases[i])
        if cls_id is None or cls_id == CLASS_ID["worker"]:
            continue

        class_name = CLASS_NAMES[cls_id]
        score = float(logits[i])

        if score < get_threshold(class_name, condition):
            continue

        bx1, by1, bx2, by2 = xyxy_arr[i]
        bx1 = max(0.0, min(float(w), float(bx1)))
        by1 = max(0.0, min(float(h), float(by1)))
        bx2 = max(0.0, min(float(w), float(bx2)))
        by2 = max(0.0, min(float(h), float(by2)))

        box_w = bx2 - bx1
        box_h = by2 - by1
        if box_w < PPE_BOX_MIN_PX or box_h < PPE_BOX_MIN_PX:
            continue

        # Association check: PPE center must be within a padded worker region
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        near_worker = any(
            (px1 <= cx <= px2 and py1 <= cy <= py2)
            for px1, py1, px2, py2 in padded_workers
        )
        if not near_worker:
            continue  # discard — not associated with any detected worker

        ppe_dets.append({
            "class_id":   cls_id,
            "class_name": class_name,
            "score":      score,
            "xyxy":       [bx1, by1, bx2, by2],
        })

    return ppe_dets


def detect_all_instances(inf_image: np.ndarray, orig_image: np.ndarray,
                         dino_model, device: str, condition: str) -> list:
    """
    FIX-01 + FIX-02 + B2: Full detection with per-class NMS.

    Pass 1: Worker detection on full image.
    Pass 2a: PPE sub-detection on each worker crop (all conditions).
    Pass 2b: Full-image PPE detection (crowded condition only — B2).
    NMS:    Applied per-class only — cross-class suppression is OFF.

    Returns list of all detections (workers + PPE) with full-image coordinates.
    """
    h, w = inf_image.shape[:2]

    # Pass 1: Worker boxes
    worker_dets = run_worker_pass(inf_image, dino_model, device, condition)

    # Pass 2a: PPE boxes per worker crop (standard pass)
    all_ppe_dets = []
    for worker_det in worker_dets:
        ppe_dets = run_ppe_pass_on_worker_crop(
            inf_image, worker_det, w, h, dino_model, device, condition
        )
        all_ppe_dets.extend(ppe_dets)

    # Pass 2b: Full-image PPE pass for any scene with clustered workers.
    # Originally crowded-only; extended to all conditions so that Normal/Dusty/Low-light
    # images with groups of workers also get the full-image sweep.
    # NMS deduplicates any overlap with Pass 2a results.
    if len(worker_dets) >= CROWDED_SCENE_THRESHOLD:
        full_img_ppe = run_ppe_pass_on_full_image(
            inf_image, worker_dets, dino_model, device, condition
        )
        all_ppe_dets.extend(full_img_ppe)

    # Combine workers + all PPE
    all_dets = worker_dets + all_ppe_dets

    # FIX-02: Per-class NMS deduplicates crop-based and full-image detections.
    # Worker boxes and PPE boxes are NEVER suppressed against each other.
    all_dets = per_class_nms(all_dets)

    return all_dets


# ─────────────────────────────────────────────────────────────────────────────
# SAM MASK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def get_sam_mask(sam_predictor, xyxy_box: list, image_source: np.ndarray):
    """
    Run SAM on a single bounding box. Returns (polygons, area).
    Falls back to bbox rectangle if SAM fails.
    """
    x1, y1, x2, y2 = xyxy_box
    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

    try:
        with torch.no_grad():
            masks, _, _ = sam_predictor.predict(
                box=box_np,
                multimask_output=False,
            )
        mask = masks[0]
        area = float(np.sum(mask))
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx  = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) >= 3:
                polygons.append(approx.flatten().tolist())
        if polygons:
            return polygons, area
    except Exception as e:
        pass

    # Fallback: rectangle from bbox
    x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
    bw = x2i - x1i
    bh = y2i - y1i
    fallback_poly = [[x1i, y1i, x2i, y1i, x2i, y2i, x1i, y2i]]
    return fallback_poly, float(bw * bh)


# ─────────────────────────────────────────────────────────────────────────────
# PER-IMAGE ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

def annotate_image(img_path: str, dino_model, sam_predictor, device: str,
                   condition: str, sequence_id: str, frame_id: int,
                   image_id: int, annotation_id_start: int,
                   next_track_id: dict, track_id_counter: list,
                   no_auto_detect: bool = False) -> tuple:
    """
    Process one image end-to-end:
      1. Load image
      2. Auto-detect scene condition (may refine folder-based condition)
      3. Apply per-condition preprocessing (CLAHE/gamma) on inference copy
      4. Two-pass detection (worker → PPE per worker)
      5. Per-class NMS
      6. SAM mask generation per instance
      7. Build COCO annotation records

    Returns: (image_record, annotations, annotation_id_end)
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w        = img_bgr.shape[:2]
    img_filename = Path(img_path).name
    synth        = is_synthetic(img_filename)

    # FIX-05: Auto-detect condition — skip if --no-auto-detect for speed
    if no_auto_detect:
        auto_condition = condition   # trust folder condition directly, save 1 DINO call
    else:
        auto_condition = detect_scene_condition(img_bgr, dino_model, device,
                                                folder_condition=condition)

    # FIX-04: Build inference copy with preprocessing (source image untouched)
    inf_image = preprocess_for_inference(img_bgr, auto_condition)

    # FIX-01 + FIX-02: Two-pass detection with per-class NMS
    all_dets = detect_all_instances(inf_image, img_bgr, dino_model, device, auto_condition)

    n_workers    = sum(1 for d in all_dets if d["class_id"] == CLASS_ID["worker"])
    crowded_scene = n_workers >= CROWDED_SCENE_THRESHOLD

    dust_zone = make_dust_zone_placeholder(w, h) if auto_condition == "dusty" else []

    # Map condition to site-level environment tag (image-level metadata only,
    # never annotated as an object, box, or mask)
    CONDITION_TO_ENVIRONMENT = {
        "normal":    "normal",
        "dusty":     "dusty",
        "low_light": "low_light",
        "crowded":   "crowded",
    }

    image_record = {
        "id":              image_id,
        "file_name":       img_filename,
        "height":          h,
        "width":           w,
        "environment":     CONDITION_TO_ENVIRONMENT.get(auto_condition, "normal"),
        "scene_condition": auto_condition,
        "crowded_scene":   crowded_scene,
        "sequence_id":     sequence_id,
        "frame_id":        frame_id,
        "dust_zone":       dust_zone,
        "is_synthetic":    synth,
        "date_captured":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # SAM setup for original image (use original for accurate masks)
    if sam_predictor is not None and len(all_dets) > 0:
        rgb_source = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            sam_predictor.set_image(rgb_source)

    annotations = []
    ann_id      = annotation_id_start

    for det in all_dets:
        cls_id     = det["class_id"]
        class_name = det["class_name"]
        score      = det["score"]
        x1, y1, x2, y2 = det["xyxy"]

        coco_bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        if coco_bbox[2] < WORKER_BOX_MIN_PX or coco_bbox[3] < WORKER_BOX_MIN_PX:
            continue

        # SAM segmentation mask
        if sam_predictor is not None:
            polygons, area = get_sam_mask(sam_predictor, det["xyxy"], rgb_source)
        else:
            x, y, bw, bh = coco_bbox
            polygons = [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]
            area     = float(bw * bh)

        flags = auto_instance_flags(score, auto_condition, n_workers, coco_bbox, w, h)

        # SAMURAI track_id: workers get a stable ID per sequence; PPE gets -1
        if cls_id == CLASS_ID["worker"]:
            x_center  = coco_bbox[0] + coco_bbox[2] // 2
            track_key = f"{sequence_id}_x{x_center // 50}"
            if track_key not in next_track_id:
                next_track_id[track_key] = track_id_counter[0]
                track_id_counter[0] += 1
            track_id = next_track_id[track_key]
        else:
            track_id = -1

        annotation = {
            "id":               ann_id,
            "image_id":         image_id,
            "category_id":      cls_id,
            "segmentation":     polygons,
            "area":             area,
            "bbox":             coco_bbox,
            "iscrowd":          0,
            "score":            score,
            "dust_occluded":    flags["dust_occluded"],
            "low_visibility":   flags["low_visibility"],
            "low_conf":         flags["low_conf"],
            "truncated":        flags["truncated"],
            "crowd_occluded":   flags["crowd_occluded"],
            "overexposed_glare":flags["overexposed_glare"],
            "no_ppe":           flags["no_ppe"],
            "no_ppe_visible":   flags["no_ppe_visible"],
            "track_id":         track_id,
            "sequence_id":      sequence_id,
            "frame_id":         frame_id,
        }
        annotations.append(annotation)
        ann_id += 1

    return image_record, annotations, ann_id


# ─────────────────────────────────────────────────────────────────────────────
# FIX-06: YOLO EXPORT — VERIFIED CLASS IDS 0–3 ONLY
# ─────────────────────────────────────────────────────────────────────────────

def write_yolo_files(image_record: dict, annotations: list, split: str):
    """
    FIX-06: Write YOLO bbox .txt and YOLO polygon .txt.
    Asserts class IDs are in [0,1,2,3] only.
    Multiple entries per image (one per annotation instance).
    """
    img_w    = image_record["width"]
    img_h    = image_record["height"]
    img_name = Path(image_record["file_name"]).stem

    bbox_path = OUTPUT_DIR / "labels"     / split / f"{img_name}.txt"
    seg_path  = OUTPUT_DIR / "labels_seg" / split / f"{img_name}.txt"

    bbox_lines = []
    seg_lines  = []

    for ann in annotations:
        cat_id = ann["category_id"]

        # FIX-06: Validate class ID is within 4-class schema
        assert 0 <= cat_id <= 3, (
            f"Invalid class ID {cat_id} in {img_name} — must be 0–3 only"
        )

        bbox = ann["bbox"]
        cx, cy, nw, nh = coco_box_to_yolo(bbox, img_w, img_h)
        bbox_lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        poly_str = polygon_to_yolo_seg(ann["segmentation"], img_w, img_h)
        if poly_str:
            seg_lines.append(f"{cat_id} {poly_str}")
        else:
            seg_lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if EXPORT_YOLO_BBOX:
        with open(bbox_path, "w") as f:
            f.write("\n".join(bbox_lines))

    if EXPORT_YOLO_SEG:
        with open(seg_path, "w") as f:
            f.write("\n".join(seg_lines))


# ─────────────────────────────────────────────────────────────────────────────
# DATA.YAML WRITER (4-class locked)
# ─────────────────────────────────────────────────────────────────────────────

def write_data_yaml():
    """
    Write 3-class data.yaml. nc=3, names indexed 0–2.
    No reserved slots. Class IDs are sequential.
    """
    yaml_content = f"""# BuildSight AI — 3-Class PPE Detection Config
# FINAL | Green Build AI | NBC 2016 Aligned
# Generated by annotate_indian_dataset.py

path: {OUTPUT_DIR.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 3
names:
  0: helmet
  1: safety_vest
  2: worker
"""
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  data.yaml -> {yaml_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_annotation_pipeline(args):
    setup_output_dirs()
    dino_model, sam_predictor, device = load_models(skip_sam=args.skip_sam)

    conditions_to_run = args.conditions.split(",") if args.conditions else CONDITIONS

    # Collect image paths per condition
    all_images_by_condition = {}
    for condition in conditions_to_run:
        folder = DATA_DIR / condition
        if not folder.exists():
            print(f"  [SKIP] Not found: {folder}")
            continue
        files = sorted([
            f.name for f in folder.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ])
        if args.test_batch:
            files = files[:args.test_batch]
        all_images_by_condition[condition] = files
        print(f"  {condition}: {len(files)} images")

    # Assign train/val/test splits
    split_map = {}
    for condition, files in all_images_by_condition.items():
        splits = split_files(files)
        for split_name, split_files_list in splits.items():
            for f in split_files_list:
                split_map[(condition, f)] = split_name

    # Initialize COCO output structures
    coco_by_split = {
        split: {
            "info": {
                "description":  "BuildSight Indian Construction Site Dataset — PPE Detection",
                "version":      "2.0",
                "year":         2025,
                "contributor":  "Green Build AI",
                "date_created": datetime.utcnow().strftime("%Y/%m/%d"),
            },
            "licenses":    [],
            "images":      [],
            "annotations": [],
            "categories":  COCO_CATEGORIES,
        }
        for split in ["train", "val", "test"]
    }

    image_id_counter  = 1
    annotation_id     = 1
    next_track_id     = {}
    track_id_counter  = [1]
    total_annotations = 0
    stats = {c: {"images": 0, "annotations": 0} for c in conditions_to_run}

    # Manifest for condition auto-detection log
    manifest_rows = []

    for condition in conditions_to_run:
        if condition not in all_images_by_condition:
            continue

        files  = all_images_by_condition[condition]
        folder = DATA_DIR / condition
        seq_assignments = cluster_into_sequences(files, gap_ms=SEQUENCE_GAP_MS)

        print(f"\n{'='*60}")
        print(f"Processing: {condition} ({len(files)} images)")
        print(f"{'='*60}")

        for img_filename in tqdm(files, desc=condition[:20], unit="img", ncols=100):
            img_path = folder / img_filename
            if not img_path.exists():
                continue

            seq_idx, frame_idx = seq_assignments.get(img_filename, (0, 0))
            sequence_id        = f"{condition[:8]}_{seq_idx:04d}"
            split              = split_map.get((condition, img_filename), "train")

            # FIX-04: pass folder-derived condition; auto-detector may refine it
            folder_cond = FOLDER_TO_CONDITION.get(condition, "normal")

            # RESUME: skip inference if YOLO label already exists; reconstruct COCO entry
            lbl_path = OUTPUT_DIR / "labels" / split / (Path(img_filename).stem + ".txt")
            dst_img_path = OUTPUT_DIR / "images" / split / img_filename
            if lbl_path.exists() and dst_img_path.exists():
                img_bgr_rs = cv2.imread(str(dst_img_path))
                rs_h, rs_w = (img_bgr_rs.shape[:2] if img_bgr_rs is not None else (0, 0))
                image_record = {
                    "id": image_id_counter, "file_name": img_filename,
                    "height": rs_h, "width": rs_w,
                    "environment": FOLDER_TO_CONDITION.get(condition, "normal"),
                    "scene_condition": folder_cond, "crowded_scene": False,
                    "sequence_id": sequence_id, "frame_id": frame_idx,
                    "dust_zone": [], "is_synthetic": False,
                    "date_captured": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                annotations = []
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cat_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    bx = (cx - bw / 2) * rs_w
                    by = (cy - bh / 2) * rs_h
                    bw_abs = bw * rs_w
                    bh_abs = bh * rs_h
                    annotations.append({
                        "id": annotation_id, "image_id": image_id_counter,
                        "category_id": cat_id,
                        "bbox": [bx, by, bw_abs, bh_abs],
                        "area": bw_abs * bh_abs,
                        "segmentation": [],
                        "iscrowd": 0, "score": 1.0,
                        "track_id": -1, "sequence_id": sequence_id, "frame_id": frame_idx,
                    })
                    annotation_id += 1
                coco_by_split[split]["images"].append(image_record)
                coco_by_split[split]["annotations"].extend(annotations)
                stats[condition]["images"]      += 1
                stats[condition]["annotations"] += len(annotations)
                image_id_counter += 1
                continue

            try:
                image_record, annotations, annotation_id = annotate_image(
                    img_path            = str(img_path),
                    dino_model          = dino_model,
                    sam_predictor       = sam_predictor,
                    device              = device,
                    condition           = folder_cond,
                    sequence_id         = sequence_id,
                    frame_id            = frame_idx,
                    image_id            = image_id_counter,
                    annotation_id_start = annotation_id,
                    next_track_id       = next_track_id,
                    track_id_counter    = track_id_counter,
                    no_auto_detect      = args.no_auto_detect,
                )
            except Exception as e:
                print(f"\n  [ERROR] {img_filename}: {e}")
                continue

            # Copy image to output
            dst_img = OUTPUT_DIR / "images" / split / img_filename
            if not dst_img.exists():
                shutil.copy2(str(img_path), str(dst_img))

            # Accumulate COCO
            coco_by_split[split]["images"].append(image_record)
            coco_by_split[split]["annotations"].extend(annotations)

            # Write YOLO files
            write_yolo_files(image_record, annotations, split)

            # Log manifest entry
            auto_cond  = image_record["scene_condition"]
            n_workers  = sum(1 for a in annotations if a["category_id"] == CLASS_ID["worker"])
            n_helmets  = sum(1 for a in annotations if a["category_id"] == CLASS_ID["helmet"])
            n_vests    = sum(1 for a in annotations if a["category_id"] == CLASS_ID["safety_vest"])
            manifest_rows.append({
                "file":      img_filename,
                "condition": auto_cond,
                "workers":   n_workers,
                "helmets":   n_helmets,
                "vests":     n_vests,
            })

            stats[condition]["images"]      += 1
            stats[condition]["annotations"] += len(annotations)
            total_annotations               += len(annotations)
            image_id_counter                += 1

    # Save COCO JSONs
    if EXPORT_COCO_JSON:
        # If running a subset of conditions, write to new_crowded_*.json
        # so existing full-dataset JSONs are not overwritten.
        # Run merge_crowded_annotations.py afterwards to integrate.
        partial_run = bool(args.conditions)
        for split, coco_data in coco_by_split.items():
            if partial_run:
                prefix = "new_crowded"
            else:
                prefix = "instances"
            json_path = OUTPUT_DIR / "annotations" / f"{prefix}_{split}.json"
            with open(json_path, "w") as f:
                json.dump(coco_data, f, indent=2)
            print(f"\n  [{split}] COCO: {len(coco_data['images'])} imgs, "
                  f"{len(coco_data['annotations'])} anns -> {json_path}")
        if partial_run:
            print("\n  Run scripts/merge_crowded_annotations.py to integrate into full dataset.")

    # Write data.yaml (4-class)
    write_data_yaml()

    # Write detection manifest
    manifest_path = OUTPUT_DIR / "detection_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_rows, f, indent=2)
    print(f"\n  Manifest -> {manifest_path}")

    # Summary
    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"{'Condition':<30} {'Images':>8} {'Annotations':>14}")
    print("-" * 55)
    for condition, s in stats.items():
        print(f"  {condition:<28} {s['images']:>8} {s['annotations']:>14}")
    print("-" * 55)
    total_imgs = sum(s["images"] for s in stats.values())
    print(f"  {'TOTAL':<28} {total_imgs:>8} {total_annotations:>14}")
    print(f"\nOutput: {OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BuildSight Indian Dataset Annotation Pipeline — Fixed v2"
    )
    parser.add_argument("--test-batch", type=int, default=0,
                        help="Process only first N images per condition (0 = all)")
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip SAM segmentation (bbox only, much faster)")
    parser.add_argument("--conditions", type=str, default="",
                        help="Comma-separated conditions to process (default: all)")
    parser.add_argument("--no-auto-detect", action="store_true", dest="no_auto_detect",
                        help="Skip FIX-05 auto-detection, use folder condition directly (faster)")
    args = parser.parse_args()

    run_annotation_pipeline(args)
