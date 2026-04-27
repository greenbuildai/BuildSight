"""
pipeline_config.py
==================
BuildSight AI — Annotation Pipeline Configuration
Green Build AI | IGBC AP | NBC 2016 Aligned

FINAL 3-CLASS SCHEMA:
  Class ID 0 → helmet       (hard_hat merged in)
  Class ID 1 → safety_vest  (high_vis_jacket merged in)
  Class ID 2 → worker

ALL thresholds and parameters live here.
DO NOT hardcode any value in the pipeline scripts.

Applies fixes:
  FIX-03 — Per-class confidence thresholds
  FIX-04 — Per-condition threshold adjustments and preprocessing params
  FIX-02 — Per-class NMS IoU thresholds
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(r"E:\Company\Green Build AI\Prototypes\BuildSight")
DATA_DIR   = BASE_DIR / "Dataset" / "Indian Dataset"
OUTPUT_DIR = BASE_DIR / "Dataset" / "Final_Annotated_Dataset"

DINO_CONFIG  = None   # resolved at runtime from groundingdino package
DINO_WEIGHTS = BASE_DIR / "weights" / "groundingdino_swint_ogc.pth"
SAM_WEIGHTS  = BASE_DIR / "weights" / "sam_vit_b_01ec64.pth"

# ─────────────────────────────────────────────────────────────────────────────
# 4-CLASS SCHEMA — FINAL, LOCKED
# Classes outside this set must not appear anywhere in the pipeline output.
# ─────────────────────────────────────────────────────────────────────────────

CLASS_ID = {
    "helmet":       0,   # includes hard_hat, yellow/white/green helmets
    "safety_vest":  1,   # includes high_vis_jacket, reflective workwear
    "worker":       2,   # full body box — lungi, dhoti, kurta, standard workwear
}

CLASS_NAMES = {v: k for k, v in CLASS_ID.items()}   # reverse lookup: int → str
NUM_CLASSES = 3

# Merged / removed classes — these are canonicalized on ingest:
#   gloves, face_mask, dust_mask, safety_goggles  → dropped
#   hard_hat, hard hat                             → helmet (class 0)
#   high_vis_jacket, high vis                      → safety_vest (class 1)

# ─────────────────────────────────────────────────────────────────────────────
# GROUNDING DINO — DETECTION PROMPTS (two-pass approach)
# ─────────────────────────────────────────────────────────────────────────────

# Pass 1: Full-image worker detection (Strictly human-centric to avoid machinery hallucinations)
WORKER_TEXT_PROMPT = "human . person . man . woman . laborer ."

# Pass 2: PPE sub-detection run on each worker crop (Ensembled synonyms for higher accuracy)
# NOTE: "vest" alone removed — too generic, causes shirts/pants false positives
PPE_TEXT_PROMPT    = "helmet . hard hat . yellow hard hat . construction helmet . reflective safety vest . fluorescent safety vest . orange safety vest . yellow safety vest . bright safety vest ."

# Fix-UP Step 3: Simplified crop prompt — tighter vocabulary, less DINO distraction
# NOTE: "vest" alone removed — too generic, causes shirts/pants false positives
PPE_CROP_TEXT_PROMPT = "helmet . hard hat . yellow hard hat . construction helmet . reflective safety vest . fluorescent safety vest . orange safety vest . yellow safety vest . bright safety vest ."

# Permissive DINO gate (per-class thresholds applied AFTER to filter output)
DINO_BOX_THRESHOLD  = 0.10   # Standard gate: workers scored 0.10+ pass through
DINO_TEXT_THRESHOLD = 0.08
DINO_CROP_BOX_THRESHOLD  = 0.06   # PPE crop gate: allows very-low-confidence helmet candidates
DINO_CROP_TEXT_THRESHOLD = 0.06

# Lower gate for worker detection in low_light — DINO often scores partially-occluded
# or dim workers at 0.08-0.09 which the standard 0.10 gate rejects entirely.
# Only used for Pass 1 worker detection; not for PPE detection.
DINO_BOX_THRESHOLD_LOWLIGHT = 0.08

# Phrase → class_id mapping for both passes
PHRASE_TO_CLASS = {
    "person":             CLASS_ID["worker"],
    "worker":             CLASS_ID["worker"],
    "construction worker":CLASS_ID["worker"],
    "man":                CLASS_ID["worker"],
    "safety helmet":      CLASS_ID["helmet"],
    "helmet":             CLASS_ID["helmet"],
    "hard hat":           CLASS_ID["helmet"],
    "hard_hat":           CLASS_ID["helmet"],
    "yellow hard hat":    CLASS_ID["helmet"],
    "construction helmet":CLASS_ID["helmet"],
    "safety vest":        CLASS_ID["safety_vest"],
    "reflective safety vest": CLASS_ID["safety_vest"],
    "reflective vest":    CLASS_ID["safety_vest"],
    # "vest" intentionally removed — too generic, DINO maps shirts/pants to it
    "safety_vest":        CLASS_ID["safety_vest"],
    "high vis":           CLASS_ID["safety_vest"],
    "high visibility":    CLASS_ID["safety_vest"],
    "high_vis":           CLASS_ID["safety_vest"],
    "high-visibility jacket": CLASS_ID["safety_vest"],
    "neon safety vest":   CLASS_ID["safety_vest"],
    "orange safety vest": CLASS_ID["safety_vest"],
    "yellow safety vest": CLASS_ID["safety_vest"],
    "orange vest":        CLASS_ID["safety_vest"],
    "construction vest":  CLASS_ID["safety_vest"],
    "orange safety clothing": CLASS_ID["safety_vest"],
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-03: PER-CLASS CONFIDENCE THRESHOLDS (base values)
# Do NOT use a single global threshold.
# Each class has its own independently configurable value.
# ─────────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLDS = {
    "helmet":       0.14,   # Lowered from 0.16: allows effective low_light threshold to reach 0.08 (=max(0.08, 0.14-0.07))
    "safety_vest":  0.15,   # Lowered from 0.20 to catch the miss in image 1
    "worker":       0.20,   # Drastically lowered from 0.40 to stop ignoring background workers
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-04: PER-CONDITION THRESHOLD DELTA
# Applied on top of CONFIDENCE_THRESHOLDS for each condition.
# ─────────────────────────────────────────────────────────────────────────────

# Per-class, per-condition deltas.
# Worker gets +0.07 in normal to block debris/tripod FPs (scored 0.256-0.269).
# Vest gets +0.00 in normal — structural guards (centre-Y, height-frac) handle FPs.
# Helmet gets +0.03 in normal — moderate lift to reduce poster FPs.
CONDITION_THRESHOLD_DELTA = {
    "normal": {
        "helmet":      +0.03,
        "safety_vest": +0.00,   # no delta: orange vests score 0.18-0.22; structural guards handle FPs
        "worker":      +0.07,   # debris/tripod scored 0.256-0.269; needs full lift to filter
    },
    "crowded": {
        "helmet":      -0.08,   # effective 0.06 (floor) — bent-over workers + partial views; push to DINO gate floor
        "safety_vest": +0.06,   # effective 0.21 — tighter to suppress plain-shirt/clothing FPs in crowded scenes
        "worker":      +0.00,
    },
    "dusty": {
        "helmet":      -0.00,
        "safety_vest": -0.00,
        "worker":      -0.00,
    },
    "low_light": {
        "helmet":      -0.07,
        "safety_vest": -0.07,
        "worker":      -0.08,   # effective threshold = max(0.06, 0.20-0.08) = 0.12
    },
}


def get_threshold(class_name: str, condition: str) -> float:
    """
    Return the effective confidence threshold for a class in a given condition.
    Applies the per-class, per-condition delta on top of the base threshold.
    Floor at 0.10 to prevent threshold collapse.
    """
    base  = CONFIDENCE_THRESHOLDS[class_name]
    delta = CONDITION_THRESHOLD_DELTA.get(condition, {}).get(class_name, 0.0)
    return max(0.06, base + delta)


# ─────────────────────────────────────────────────────────────────────────────
# FIX-02: PER-CLASS NMS IoU THRESHOLDS
# Cross-class NMS is DISABLED.  Only boxes of the SAME class suppress each other.
# ─────────────────────────────────────────────────────────────────────────────

NMS_IOU_THRESHOLD = {
    "helmet":       0.30,   # tight: post-processing cluster-merge catches IoU 0.20-0.30; NMS removes 0.30+
    "safety_vest":  0.35,   # cluster-merge handles dedup; NMS only removes truly identical/heavily overlapping boxes
    "worker":       0.40,   # worker dedup NMS handles 0.40-0.50 range with separate containment check
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-04: CLAHE PARAMETERS PER CONDITION
# Applied to a preprocessing copy of the image only.
# Source images are NEVER modified.
# ─────────────────────────────────────────────────────────────────────────────

CLAHE_PARAMS = {
    "normal":    None,                                         # no CLAHE
    "crowded":   None,                                         # no CLAHE
    "dusty":     {"clip_limit": 3.0, "tile_grid_size": (8, 8)},
    "low_light": {"clip_limit": 4.0, "tile_grid_size": (8, 8)},
}

# FIX-04: GAMMA CORRECTION PER CONDITION (applied after CLAHE if enabled)
GAMMA_CORRECTION = {
    "normal":    1.0,    # no correction
    "crowded":   1.0,    # no correction
    "dusty":     1.0,    # CLAHE only — gamma not needed for dust
    "low_light": 1.8,    # brighten dark frames significantly
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-05: CONDITION AUTO-DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60    # mean gray < 60 → low_light
DUSTY_STD_THRESHOLD            = 45    # std gray < 45 → possibly dusty
DUSTY_SATURATION_THRESHOLD     = 60    # mean HSV-S < 60 → dusty (low color)
CROWDED_WORKER_COUNT_THRESHOLD = 5     # ≥5 workers detected → crowded
CROWDED_QUICK_DETECT_THRESHOLD = 0.25  # worker detection threshold for quick-pass

# Alias — used by annotate_indian_dataset.py
CROWDED_SCENE_THRESHOLD = CROWDED_WORKER_COUNT_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# FIX-01: PER-INSTANCE PPE SEPARATION — WORKER CROP SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Asymmetric crop padding — each direction tuned independently:
#   H  (left/right): tight (10%) to stay close to the detected worker and avoid
#                    capturing adjacent workers' PPE in the horizontal padding zone.
#   V_UP (above):    wide (40%) to capture heads/helmets that sit above the box top
#                    because DINO's worker box often starts at shoulder level.
#   V_DN (below):    tight (10%) — feet rarely have PPE to detect.
# PPE_CONTAINMENT_TOL_H and PPE_CONTAINMENT_TOL_V_UP must stay ≥ these values
# so any PPE found in the padded crop passes the subsequent containment check.
WORKER_CROP_PAD_H    = 0.10   # horizontal (left + right)
WORKER_CROP_PAD_V_UP = 0.40   # vertical upward  (above box top)
WORKER_CROP_PAD_V_DN = 0.10   # vertical downward (below box bottom)

# Legacy alias — used by full-image PPE pass association; keep at 0.20.
WORKER_CROP_PADDING  = 0.20

# Fix-UP Step 2: upscale worker crops before feeding to DINO — magnifying glass effect
# Increased to 2.5x for superior detail on far-away head protection
WORKER_CROP_UPSCALE_FACTOR = 3.0   # Reverted from 4.0x — 4.0x caused thermal throttling on RTX 4050

# Minimum worker box height (pixels) below which PPE sub-detection is skipped
# (far-away workers are too small to reliably detect individual PPE items)
WORKER_MIN_HEIGHT_FOR_PPE = 40   # lowered from 50: upscaling makes smaller workers viable

# Minimum PPE box size (pixels) — discard micro-boxes
PPE_BOX_MIN_PX = 8

# Worker box minimum size to include in output at all
WORKER_BOX_MIN_PX = 12

# Anti-pictogram / anti-safety-sign filter (Pass 1 worker detection)
# Safety signs depict human figures in flat uniform colors (low texture).
# Real workers have complex texture from clothing patterns, skin, and shadows.
# Reject worker crops whose colour std-dev falls below this value.
WORKER_MIN_COLOR_STD = 15.0

# Helmet aspect ratio bounds — real helmet is roughly square
# Shirt collar / wide objects fall outside this range and are rejected
HELMET_MIN_ASPECT_RATIO = 0.35  # width/height — narrower than this = not a helmet (relaxed for side-profile views)
HELMET_MAX_ASPECT_RATIO = 1.8   # width/height — wider than this = shirt/broad object

# Helmet must sit in the TOP of the worker box — shirts/chests are lower.
# PPE centre-Y must be within the top 45% of the worker bounding box height.
# Allows helmets that protrude slightly above the box top (handled by V_UP tolerance).
HELMET_MAX_CENTRE_Y_FRACTION = 0.45

# Vest height constraint — real vest covers torso only
# Vest height constraint only applied to close-up workers (large worker boxes).
# full-body clothing false positives only occur on close-up workers (bh > 40px).
# Torso typically ends at ~60-65% of person height. Anything more is pants.
VEST_BOX_MAX_HEIGHT_FRACTION    = 0.75   # raised from 0.55: close-up vests span ~60-70% of worker box; pants handled by centre-Y guard
VEST_WORKER_MIN_H_FOR_CHECK     = 30     # lowered from 40: apply check to more workers to catch more false positives

# Vest vertical centre position guard
# A real safety vest is worn on the torso — its centre Y must be in the
# upper 65% of the worker box. Pants/trousers have their centre in the
# lower portion (65-100%) and are rejected by this constraint.
# Plain shirts that pass the height fraction check are also rejected because
# they typically span from ~10% to ~50% of the worker box; only high-vis vests
# with high DINO confidence (>= 0.28) survive all guards together.
VEST_MAX_CENTRE_Y_FRACTION = 0.70   # vest centre must be within top 70% of worker box height (pants centre sits at 75-85%)

# PPE Containment Check tolerances (fraction of worker box dimension)
# After remapping PPE coords back to full-image space, the PPE centre must lie
# within the original worker box extended by these margins.
# Rejects vest/helmet detections that land on adjacent signs in the padding zone.
PPE_CONTAINMENT_TOL_H      = 0.10   # Relaxed to 0.10 to capture side-profile vests
PPE_CONTAINMENT_TOL_V_UP   = 0.40   # +40% of worker height upward — covers heads above box top
#                                     (crop padding is 0.35 so anything in the padded crop passes)
PPE_CONTAINMENT_TOL_V_DOWN = 0.30   # +30% of worker height downward — covers bent-over workers whose head is at box bottom

# MAXIMUM worker box size (as fraction of image height/width)
# Set to 0.98 to keep close-up workers (filling most of frame) while still
# blocking full-frame machinery (cranes/excavators are typically > 90% of frame).
WORKER_BOX_MAX_FRACTION = 0.98

# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE FLAGS THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

LOW_CONF_THRESHOLD = 0.35   # score below this → low_conf flag set

# ─────────────────────────────────────────────────────────────────────────────
# SAMURAI / SEQUENCE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_GAP_MS = 500   # images within 500ms → same SAMURAI sequence

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT FLAGS
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_YOLO_BBOX = True    # write YOLO bbox .txt files
EXPORT_YOLO_SEG  = True    # write YOLO polygon segmentation .txt files
EXPORT_COCO_JSON = True    # write COCO JSON annotation files


# ─────────────────────────────────────────────────────────────────────────────
# FIX-04: PER-CONDITION THRESHOLD DELTA
# Applied on top of CONFIDENCE_THRESHOLDS for each condition.
# ─────────────────────────────────────────────────────────────────────────────

# Per-class, per-condition deltas.
# Worker gets +0.07 in normal to block debris/tripod FPs (scored 0.256-0.269).
# Vest gets +0.00 in normal — structural guards (centre-Y, height-frac) handle FPs.
# Helmet gets +0.03 in normal — moderate lift to reduce poster FPs.
CONDITION_THRESHOLD_DELTA = {
    "normal": {
        "helmet":      +0.03,
        "safety_vest": +0.00,   # no delta: orange vests score 0.18-0.22; structural guards handle FPs
        "worker":      +0.07,   # debris/tripod scored 0.256-0.269; needs full lift to filter
    },
    "crowded": {
        "helmet":      -0.08,   # effective 0.06 (floor) — bent-over workers + partial views; push to DINO gate floor
        "safety_vest": +0.06,   # effective 0.21 — tighter to suppress plain-shirt/clothing FPs in crowded scenes
        "worker":      +0.00,
    },
    "dusty": {
        "helmet":      -0.00,
        "safety_vest": -0.00,
        "worker":      -0.00,
    },
    "low_light": {
        "helmet":      -0.07,
        "safety_vest": -0.07,
        "worker":      -0.08,   # effective threshold = max(0.06, 0.20-0.08) = 0.12
    },
}


def get_threshold(class_name: str, condition: str) -> float:
    """
    Return the effective confidence threshold for a class in a given condition.
    Applies the per-class, per-condition delta on top of the base threshold.
    Floor at 0.10 to prevent threshold collapse.
    """
    base  = CONFIDENCE_THRESHOLDS[class_name]
    delta = CONDITION_THRESHOLD_DELTA.get(condition, {}).get(class_name, 0.0)
    return max(0.06, base + delta)


# ─────────────────────────────────────────────────────────────────────────────
# FIX-02: PER-CLASS NMS IoU THRESHOLDS
# Cross-class NMS is DISABLED.  Only boxes of the SAME class suppress each other.
# ─────────────────────────────────────────────────────────────────────────────

NMS_IOU_THRESHOLD = {
    "helmet":       0.30,   # tight: post-processing cluster-merge catches IoU 0.20-0.30; NMS removes 0.30+
    "safety_vest":  0.35,   # cluster-merge handles dedup; NMS only removes truly identical/heavily overlapping boxes
    "worker":       0.40,   # worker dedup NMS handles 0.40-0.50 range with separate containment check
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-04: CLAHE PARAMETERS PER CONDITION
# Applied to a preprocessing copy of the image only.
# Source images are NEVER modified.
# ─────────────────────────────────────────────────────────────────────────────

CLAHE_PARAMS = {
    "normal":    None,                                         # no CLAHE
    "crowded":   None,                                         # no CLAHE
    "dusty":     {"clip_limit": 3.0, "tile_grid_size": (8, 8)},
    "low_light": {"clip_limit": 4.0, "tile_grid_size": (8, 8)},
}

# FIX-04: GAMMA CORRECTION PER CONDITION (applied after CLAHE if enabled)
GAMMA_CORRECTION = {
    "normal":    1.0,    # no correction
    "crowded":   1.0,    # no correction
    "dusty":     1.0,    # CLAHE only — gamma not needed for dust
    "low_light": 1.8,    # brighten dark frames significantly
}

# ─────────────────────────────────────────────────────────────────────────────
# FIX-05: CONDITION AUTO-DETECTION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

LOW_LIGHT_BRIGHTNESS_THRESHOLD = 60    # mean gray < 60 → low_light
DUSTY_STD_THRESHOLD            = 45    # std gray < 45 → possibly dusty
DUSTY_SATURATION_THRESHOLD     = 60    # mean HSV-S < 60 → dusty (low color)
CROWDED_WORKER_COUNT_THRESHOLD = 5     # ≥5 workers detected → crowded
CROWDED_QUICK_DETECT_THRESHOLD = 0.25  # worker detection threshold for quick-pass

# Alias — used by annotate_indian_dataset.py
CROWDED_SCENE_THRESHOLD = CROWDED_WORKER_COUNT_THRESHOLD

# ─────────────────────────────────────────────────────────────────────────────
# FIX-01: PER-INSTANCE PPE SEPARATION — WORKER CROP SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

# Asymmetric crop padding — each direction tuned independently:
#   H  (left/right): tight (10%) to stay close to the detected worker and avoid
#                    capturing adjacent workers' PPE in the horizontal padding zone.
#   V_UP (above):    wide (40%) to capture heads/helmets that sit above the box top
#                    because DINO's worker box often starts at shoulder level.
#   V_DN (below):    tight (10%) — feet rarely have PPE to detect.
# PPE_CONTAINMENT_TOL_H and PPE_CONTAINMENT_TOL_V_UP must stay ≥ these values
# so any PPE found in the padded crop passes the subsequent containment check.
WORKER_CROP_PAD_H    = 0.10   # horizontal (left + right)
WORKER_CROP_PAD_V_UP = 0.40   # vertical upward  (above box top)
WORKER_CROP_PAD_V_DN = 0.10   # vertical downward (below box bottom)

# Legacy alias — used by full-image PPE pass association; keep at 0.20.
WORKER_CROP_PADDING  = 0.20

# Fix-UP Step 2: upscale worker crops before feeding to DINO — magnifying glass effect
# Increased to 2.5x for superior detail on far-away head protection
WORKER_CROP_UPSCALE_FACTOR = 3.0   # Reverted from 4.0x — 4.0x caused thermal throttling on RTX 4050

# Minimum worker box height (pixels) below which PPE sub-detection is skipped
# (far-away workers are too small to reliably detect individual PPE items)
WORKER_MIN_HEIGHT_FOR_PPE = 40   # lowered from 50: upscaling makes smaller workers viable

# Minimum PPE box size (pixels) — discard micro-boxes
PPE_BOX_MIN_PX = 8

# Worker box minimum size to include in output at all
WORKER_BOX_MIN_PX = 12

# Anti-pictogram / anti-safety-sign filter (Pass 1 worker detection)
# Safety signs depict human figures in flat uniform colors (low texture).
# Real workers have complex texture from clothing patterns, skin, and shadows.
# Reject worker crops whose colour std-dev falls below this value.
WORKER_MIN_COLOR_STD = 15.0

# Helmet aspect ratio bounds — real helmet is roughly square
# Shirt collar / wide objects fall outside this range and are rejected
HELMET_MIN_ASPECT_RATIO = 0.35  # width/height — narrower than this = not a helmet (relaxed for side-profile views)
HELMET_MAX_ASPECT_RATIO = 1.8   # width/height — wider than this = shirt/broad object

# Helmet must sit in the TOP of the worker box — shirts/chests are lower.
# PPE centre-Y must be within the top 45% of the worker bounding box height.
# Allows helmets that protrude slightly above the box top (handled by V_UP tolerance).
HELMET_MAX_CENTRE_Y_FRACTION = 0.45

# Vest height constraint — real vest covers torso only
# Vest height constraint only applied to close-up workers (large worker boxes).
# full-body clothing false positives only occur on close-up workers (bh > 40px).
# Torso typically ends at ~60-65% of person height. Anything more is pants.
VEST_BOX_MAX_HEIGHT_FRACTION    = 0.75   # raised from 0.55: close-up vests span ~60-70% of worker box; pants handled by centre-Y guard
VEST_WORKER_MIN_H_FOR_CHECK     = 30     # lowered from 40: apply check to more workers to catch more false positives

# Vest vertical centre position guard
# A real safety vest is worn on the torso — its centre Y must be in the
# upper 65% of the worker box. Pants/trousers have their centre in the
# lower portion (65-100%) and are rejected by this constraint.
# Plain shirts that pass the height fraction check are also rejected because
# they typically span from ~10% to ~50% of the worker box; only high-vis vests
# with high DINO confidence (>= 0.28) survive all guards together.
VEST_MAX_CENTRE_Y_FRACTION = 0.70   # vest centre must be within top 70% of worker box height (pants centre sits at 75-85%)

# PPE Containment Check tolerances (fraction of worker box dimension)
# After remapping PPE coords back to full-image space, the PPE centre must lie
# within the original worker box extended by these margins.
# Rejects vest/helmet detections that land on adjacent signs in the padding zone.
PPE_CONTAINMENT_TOL_H      = 0.10   # Relaxed to 0.10 to capture side-profile vests
PPE_CONTAINMENT_TOL_V_UP   = 0.40   # +40% of worker height upward — covers heads above box top
#                                     (crop padding is 0.35 so anything in the padded crop passes)
PPE_CONTAINMENT_TOL_V_DOWN = 0.30   # +30% of worker height downward — covers bent-over workers whose head is at box bottom

# MAXIMUM worker box size (as fraction of image height/width)
# Set to 0.98 to keep close-up workers (filling most of frame) while still
# blocking full-frame machinery (cranes/excavators are typically > 90% of frame).
WORKER_BOX_MAX_FRACTION = 0.98

# ─────────────────────────────────────────────────────────────────────────────
# INSTANCE FLAGS THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

LOW_CONF_THRESHOLD = 0.35   # score below this → low_conf flag set

# ─────────────────────────────────────────────────────────────────────────────
# SAMURAI / SEQUENCE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

SEQUENCE_GAP_MS = 500   # images within 500ms → same SAMURAI sequence

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT FLAGS
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_YOLO_BBOX = True    # write YOLO bbox .txt files
EXPORT_YOLO_SEG  = True    # write YOLO polygon segmentation .txt files
EXPORT_COCO_JSON = True    # write COCO JSON annotation files

# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Subfolder name → canonical condition key
FOLDER_TO_CONDITION = {
    "Normal_Site_Condition": "normal",
    "Dusty_Condition":       "dusty",
    "Low_Light_Condition":   "low_light",
    "Crowded_Condition":     "crowded",
}

import os

# --- GEMINI AUDITOR CONFIGURATION ---
USE_GEMINI_AUDITOR = True
GEMINI_API_KEY = "AIzaSyAozS3xFiIsqJ1wJi8WfqArxG_PcztxYQ8"
GEMINI_MODEL = "gemini-2.5-flash"

# --- CORE SETTINGS ---
dataset_dir = "Dataset/Indian Dataset/"

# All four conditions in processing order
CONDITIONS = [
    "Normal_Site_Condition",
    "Dusty_Condition",
    "Low_Light_Condition",
    "Crowded_Condition",
]
