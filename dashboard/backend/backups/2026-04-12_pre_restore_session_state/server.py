#!/usr/bin/env python3
"""
BuildSight Dashboard Backend
=============================
FastAPI server wrapping the multi-model ensemble pipeline.

Endpoints:
  GET  /api/health          — status + model/device info
  POST /api/detect/image    — PPE detection on an uploaded image
  POST /api/detect/frame    — PPE detection on a base64 video/webcam frame
  POST /api/detect/video    — full-video batch inference → download annotated MP4

Performance notes (Jovi handoff 2026-04-06):
  • Models run on GPU (RTX 4050) with FP16 mixed precision → ~6-8× faster than CPU
  • Inference is wrapped in run_in_threadpool so async endpoints never block the
    FastAPI event loop — concurrent requests stay responsive
  • Worker-to-PPE spatial association added: each worker detection now carries
    has_helmet / has_vest flags based on bounding-box overlap geometry
"""

import base64
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import BaseModel
import requests as _requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
import database
from geoai import geoai_router

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent   # BuildSight root
BACKEND_DIR = Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from adaptive_postprocess import MaterialSuppressionLayer, ValidWorkerValidator


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


for _dotenv_path in (ROOT / ".env", BACKEND_DIR / ".env"):
    _load_dotenv_file(_dotenv_path)


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value else default


MODEL_DIR = _env_path("BUILDSIGHT_MODEL_DIR", ROOT / "weights")
SASTRA_V11 = _env_path("BUILDSIGHT_MODEL_V11", MODEL_DIR / "yolov11_buildsight_best.pt")
SASTRA_V26 = _env_path("BUILDSIGHT_MODEL_V26", MODEL_DIR / "yolov26_buildsight_best.pt")

LOCAL_BEST = _env_path(
    "BUILDSIGHT_LOCAL_MODEL",
    ROOT / "buildsight-base" / "basic yolo model" /
    "output" / "kaggle_working_all_outputs" /
    "kaggle" / "working" / "runs" / "train" / "weights" / "best.pt",
)

RUNTIME_DIR = _env_path("BUILDSIGHT_RUNTIME_DIR", ROOT / "runtime")
TMP_DIR = RUNTIME_DIR / "tmp"
VIDEO_OUTPUT_DIR = RUNTIME_DIR / "video_outputs"

for _path in (RUNTIME_DIR, TMP_DIR, VIDEO_OUTPUT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

# ── device selection ───────────────────────────────────────────────────────────
# Use CUDA if available (RTX 4050 confirmed); fall back to CPU gracefully.
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_HALF = DEVICE.startswith("cuda")   # FP16 only on GPU
BACKEND_HOST = os.environ.get("BUILDSIGHT_HOST", "0.0.0.0")
BACKEND_PORT = _env_int("BUILDSIGHT_PORT", 8000)
BACKEND_LOG_LEVEL = os.environ.get("BUILDSIGHT_LOG_LEVEL", "info")

# ── class schema ───────────────────────────────────────────────────────────────
SASTRA_CLASSES = {0: "helmet", 1: "safety_vest", 2: "worker"}
LOCAL_CLASSES  = {0: "worker", 1: "helmet",      2: "safety_vest"}
COLORS = {
    "helmet":      (0,   255,   0),    # Green  (BGR)
    "safety_vest": (0,   221, 255),    # Yellow (BGR) — #ffdd00 in RGB
    "worker":      (255, 136,   0),    # Blue   (BGR)
}

# ── WBF / NMS params ───────────────────────────────────────────────────────────
# Class IDs in SASTRA schema: 0=helmet, 1=safety_vest, 2=worker
MODEL_WEIGHTS   = [0.55, 0.45]

# WBF IoU thresholds — HIGHER = fewer fusions (boxes must overlap more to be
# merged). Worker raised to 0.65 so nearby-but-different workers (typical IoU
# 0.20-0.45) are NOT fused, while same-worker outputs from two models (typical
# IoU 0.75+) still merge correctly.
WBF_IOU         = {0: 0.45, 1: 0.50, 2: 0.65}

# Post-WBF confidence gates. Helmet raised (0.20→0.30) to suppress false
# positives on hair/shoulders; vest lowered (0.22→0.14) so partially-occluded
# vests that score weakly still survive; worker kept low (0.18) to catch small
# or distant workers.
POST_WBF_GLOBAL = {0: 0.30, 1: 0.14, 2: 0.18}

# Lower pre-WBF conf so weak-but-real detections (small workers, partially
# occluded helmets) reach the WBF stage where the two models can reinforce them.
PRE_CONF        = 0.20

# NMS IoU inside each model's predict() — higher = less aggressive suppression
# so adjacent workers are NOT suppressed before WBF even sees them.
NMS_IOU         = 0.60

EARLY_EXIT_CONF = 0.75   # raised: real scenes with PPE rarely all-high-conf

# Production-calibrated defaults — never modify this dict; it is the restore target
THRESHOLD_DEFAULTS: dict[str, float] = {
    "global_floor": 0.35,   # floor for all classes
    "worker":       0.40,   # 40% reliable worker baseline for Indian CCTV sites
    "helmet":       0.45,   # 45% helmet — smaller object, needs higher bar
    "vest":         0.35,   # 35% vest — strong colour signature, safe at 35%
}

# Absolute per-class minimums — warn operators if they go below these
THRESHOLD_ABSOLUTE_MINIMUMS: dict[str, float] = {
    "worker": 0.15,
    "helmet": 0.20,
    "vest":   0.15,
}

# Live mutable state — initialised from production defaults
THRESHOLD_STATE: dict[str, float] = dict(THRESHOLD_DEFAULTS)


def update_thresholds(
    global_floor: float | None = None,
    worker: float | None = None,
    helmet: float | None = None,
    vest: float | None = None,
) -> tuple[dict[str, float], list[str]]:
    """
    Update threshold state while enforcing the master-floor relationship.
    Returns (updated_state, warnings).  Warnings are non-empty when any class
    is set below its THRESHOLD_ABSOLUTE_MINIMUMS value.
    """
    warnings: list[str] = []

    if global_floor is not None:
        THRESHOLD_STATE["global_floor"] = max(float(global_floor), 0.05)
        for cls in ("worker", "helmet", "vest"):
            THRESHOLD_STATE[cls] = max(THRESHOLD_STATE[cls], THRESHOLD_STATE["global_floor"])

    for cls, val in (("worker", worker), ("helmet", helmet), ("vest", vest)):
        if val is not None:
            clamped = max(float(val), THRESHOLD_STATE["global_floor"])
            if clamped < THRESHOLD_ABSOLUTE_MINIMUMS[cls]:
                pct = lambda v: f"{v:.0%}"
                warnings.append(
                    f"{cls.upper()} threshold {pct(clamped)} is below the recommended "
                    f"minimum {pct(THRESHOLD_ABSOLUTE_MINIMUMS[cls])} — expect increased false positives."
                )
            THRESHOLD_STATE[cls] = clamped

    return dict(THRESHOLD_STATE), warnings


def reset_thresholds_to_defaults() -> dict[str, float]:
    """Restore all thresholds to production-calibrated defaults."""
    THRESHOLD_STATE.update(THRESHOLD_DEFAULTS)
    _scene_tracker.invalidate_cache()
    return dict(THRESHOLD_STATE)

# ── Condition-specific post-WBF confidence gates ───────────────────────────────
# In low-light and dusty scenes every class scores lower, so we relax the gates
# rather than miss real detections. Crowded scenes get a slightly lower worker
# gate so distant/small workers still pass.
# Keys: SASTRA class IDs  0=helmet, 1=safety_vest, 2=worker
POST_WBF_BY_CONDITION: dict[str, dict[int, float]] = {
    "S1_normal":    {0: 0.30, 1: 0.14, 2: 0.18},
    "S2_dusty":     {0: 0.25, 1: 0.10, 2: 0.15},  # vest hard to see in dust
    "S3_low_light": {0: 0.22, 1: 0.10, 2: 0.14},  # everything scores weaker
    # RECALL FIX 2026-04-11: S4 post-WBF gates lowered for worker recall.
    # worker (cls 2) 0.20→0.14 — distant workers from two models fuse to ~0.15;
    #   clutter is still gated by is_valid_worker geometry checks.
    # helmet (cls 0) 0.28→0.18 — small helmets on distant workers score lower.
    # vest (cls 1)   0.12→0.10 — partially-visible vests in crowd score lower.
    "S4_crowded":   {0: 0.18, 1: 0.10, 2: 0.14},
}

# ── Auto Scene Classification ──────────────────────────────────────────────────
#
# Lightweight scene classifier that runs a single-model quick-pass to count
# rough workers, then classifies the condition from brightness + haze + count.
# A hysteresis tracker prevents rapid flickering between modes.
#
# Classification hierarchy (highest priority first):
#   S3_low_light : mean gray < LOW_LIGHT_THRESH
#   S2_dusty     : std gray < DUSTY_STD_THRESH AND mean sat < DUSTY_SAT_THRESH
#   S4_crowded   : rough worker count ≥ CROWD_WORKER_THRESH  OR  ≥3 workers with
#                  mean pairwise IoU ≥ CROWD_OVERLAP_THRESH
#   S1_normal    : default
#
# Hysteresis:
#   • Entering S4_crowded requires CROWD_ENTER_FRAMES consecutive S4 readings
#   • Exiting S4_crowded requires CROWD_EXIT_FRAMES consecutive non-S4 readings
#   • Other conditions: smooth over a rolling window of STABILITY_WINDOW frames

AUTO_LOW_LIGHT_THRESH = 70      # mean gray below this → S3_low_light
AUTO_DUSTY_STD_THRESH  = 48     # gray std below this → candidate dusty
AUTO_DUSTY_SAT_THRESH  = 72     # mean HSV-S below this → confirm dusty
AUTO_CROWD_WORKER_THRESH = 6    # Raised from 4->6 per user preference
AUTO_CROWD_OVERLAP_THRESH = 0.05
AUTO_QUICK_WORKER_CONF = 0.22

CROWD_ENTER_FRAMES = 2   # Lowered 3->2 for faster response
CROWD_EXIT_FRAMES  = 3   # Lowered 5->3 for faster response
STABILITY_WINDOW   = 5

# Scene classification caching — how many inference frames to skip between
# full model-based reclassifications.  Image-stats (brightness/saturation) are
# still checked every frame; only the expensive model forward-pass is skipped.
SCENE_CACHE_FRAMES = 12


def _scene_condition_to_runtime(scene_condition: str) -> str:
    return {
        "S1_NORMAL": "S1_normal",
        "S2_DUSTY": "S2_dusty",
        "S3_LOW_LIGHT": "S3_low_light",
        "S4_CROWDED": "S4_crowded",
    }.get(scene_condition, "S1_normal")


class SceneConditionTracker:
    """
    Stateful hysteresis tracker for scene conditions.

    Keeps a rolling history of raw condition readings and applies hysteresis
    rules to prevent rapid flickering between modes in borderline scenes.

    update() accepts a raw_condition string from classify_scene_fast() and
    returns the stabilised condition string. This image-based approach works
    correctly on the first frame — no temporal tracking warmup required.
    """

    def __init__(self):
        self.current: str = "S1_normal"
        self._history: list[str] = []
        self._crowd_consecutive: int = 0
        self._non_crowd_consecutive: int = 0
        self._cache_frame_count: int = 0
        self._last_raw_cond: str = "S1_normal"
        # expose current_condition alias so /api/thresholds can read it
        self.cache_invalidated: bool = False

    @property
    def current_condition(self) -> str:
        return self.current

    def invalidate_cache(self):
        """Called on threshold change — forces a model re-pass next frame."""
        self.cache_invalidated = True
        self._cache_frame_count = 0

    def update(self, raw_condition: str) -> str:
        """Feed a raw single-frame condition reading, return the stabilised condition."""
        self._history.append(raw_condition)
        if len(self._history) > max(CROWD_ENTER_FRAMES, STABILITY_WINDOW, CROWD_EXIT_FRAMES) + 2:
            self._history.pop(0)

        is_crowd_signal = (raw_condition == "S4_crowded")

        if is_crowd_signal:
            self._crowd_consecutive += 1
            self._non_crowd_consecutive = 0
        else:
            self._non_crowd_consecutive += 1
            self._crowd_consecutive = max(0, self._crowd_consecutive - 1)

        if self.current == "S4_crowded":
            if self._non_crowd_consecutive >= CROWD_EXIT_FRAMES:
                self.current = self._stabilise_non_crowd()
        else:
            if self._crowd_consecutive >= CROWD_ENTER_FRAMES:
                self.current = "S4_crowded"
                self._non_crowd_consecutive = 0
            else:
                self.current = self._stabilise_non_crowd()

        return self.current

    def _stabilise_non_crowd(self) -> str:
        recent = [c for c in self._history[-STABILITY_WINDOW:] if c != "S4_crowded"]
        if not recent:
            return "S1_normal"
        counts: dict[str, int] = {}
        for c in recent:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=lambda k: counts[k])

    def should_reclassify(self) -> bool:
        """
        Return True when the expensive model forward-pass should run.
        Image-stats conditions bypass this gate entirely.
        Invalidation (from threshold change) forces immediate re-evaluation.
        """
        if self.cache_invalidated:
            self.cache_invalidated = False
            return True
        self._cache_frame_count += 1
        return (self._cache_frame_count % SCENE_CACHE_FRAMES) == 1

    def reset(self):
        self.__init__()


# Module-level tracker instance (shared across /detect/frame calls for live mode)
_scene_tracker = SceneConditionTracker()
material_suppressor = MaterialSuppressionLayer(
    min_worker_height_px=40,
    max_worker_area_ratio=0.15,
    static_frame_threshold=60,   # 60 frames ≈ 5s at 12fps before suppression
)
worker_validator = ValidWorkerValidator(
    min_human_score=0.35,
    min_height_px=40,
    min_aspect_ratio=0.4,
    max_aspect_ratio=4.5,
    static_disqualify_frames=90,   # 90 frames ≈ 8s — workers stand still routinely
    temporal_persistence_frames=2, # accept after 2 frames, not 3
)

# Frame counter for periodic pipeline state cleanup
_inference_frame_count: int = 0


class WorkerTrackState:
    def __init__(self):
        self.tracks = {}
        self.next_track_id = 1

    def reset(self):
        self.tracks = {}
        self.next_track_id = 1

    def annotate(self, detections):
        matched_track_ids = set()
        annotated = []

        for det in detections:
            box = det["box"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0

            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                overlap = iou_box(track["box"], box)
                if overlap > 0.30 and overlap > best_iou:
                    best_iou = overlap
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1
                prev_center = (cx, cy)
                static_count = 0
                age = 1
            else:
                track = self.tracks[best_track_id]
                prev_center = track["center"]
                movement = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                static_count = track["static_count"] + 1 if movement < 8.0 else 0
                age = track["age"] + 1

            self.tracks[best_track_id] = {
                "box": list(box),
                "center": (cx, cy),
                "static_count": static_count,
                "misses": 0,
                "age": age,
            }
            matched_track_ids.add(best_track_id)

            updated = dict(det)
            updated["track_id"] = best_track_id
            # is_static only after 60 consecutive frames of < 8px movement
            # (~5 seconds at 12fps). Workers stand still constantly while working.
            updated["is_static"] = static_count >= 60

            # ── Multi-signal decay gate (cement bag / scaffold artifact suppression) ──
            # Gradual confidence decay only fires when static AND ≥2 secondary signals
            # indicate the detection is likely an artifact rather than a real worker.
            base_score = updated.get("score", 0.0)
            if updated["is_static"]:
                decay_start = 60
                decay_end = 150
                if static_count > decay_start:
                    # Secondary signals (each contributes 1 vote toward decay gate)
                    has_ppe = updated.get("has_helmet", False) or updated.get("has_vest", False)
                    suspicious_signals = 0

                    # Signal 1: unusual aspect ratio for a standing worker
                    bx1, by1, bx2, by2 = box[0], box[1], box[2], box[3]
                    bw = max(bx2 - bx1, 1.0)
                    bh = max(by2 - by1, 1.0)
                    aspect = bh / bw  # tall/narrow = real worker; squat/square = object
                    if aspect < 0.80 or aspect > 4.0:  # outside normal worker shape
                        suspicious_signals += 1

                    # Signal 2: single-model detection (low ensemble agreement)
                    if updated.get("n_models", 2) < 2:
                        suspicious_signals += 1

                    # Signal 3: low base confidence (borderline detection)
                    if base_score < 0.35:
                        suspicious_signals += 1

                    # Signal 4: no PPE seen at all (real workers almost always have some PPE)
                    if not has_ppe:
                        suspicious_signals += 1

                    # Decay only when ≥2 signals fire; PPE workers get a higher floor
                    if suspicious_signals >= 2:
                        factor = max(0.0, 1.0 - (static_count - decay_start) / (decay_end - decay_start))
                        floor = 0.15 if has_ppe else 0.02
                        updated["score"] = max(floor, base_score * factor)

            updated["confidence"] = updated.get("score", 0.0)
            annotated.append(updated)

        stale = []
        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids:
                continue
            track["misses"] += 1
            if track["misses"] > 15:
                stale.append(track_id)
        for track_id in stale:
            self.tracks.pop(track_id, None)

        return annotated


worker_track_state = WorkerTrackState()


def compute_brightness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_haze(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    contrast_score = float(np.clip((55.0 - contrast) / 55.0, 0.0, 1.0))
    saturation_score = float(np.clip((80.0 - saturation) / 80.0, 0.0, 1.0))
    return float(np.clip(0.6 * contrast_score + 0.4 * saturation_score, 0.0, 1.0))


def compute_has_ppe(worker_det: dict, helmet_detections: list, vest_detections: list) -> bool:
    box = worker_det["box"]
    return _has_nearby_ppe(box, helmet_detections + vest_detections)


def classify_scene_fast(
    img_bgr: np.ndarray,
    model,
    device: str,
    half: bool,
) -> str:
    """
    Classify a single frame into S1/S2/S3/S4 in ~1-2ms extra overhead.

    Steps:
    1. Measure brightness, contrast (std), saturation from image statistics.
    2. Run a single-model quick-pass at high pre_conf (0.22) to count rough workers.
    3. Compute mean pairwise IoU of detected workers to detect clustering.
    4. Apply priority rules: S3 > S2 > S4 > S1.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    brightness  = float(np.mean(gray))
    contrast    = float(np.std(gray))
    saturation  = float(np.mean(hsv[:, :, 1]))

    # Image-stats-only conditions (no model pass needed)
    if brightness < AUTO_LOW_LIGHT_THRESH:
        return "S3_low_light"
    if contrast < AUTO_DUSTY_STD_THRESH and saturation < AUTO_DUSTY_SAT_THRESH:
        return "S2_dusty"

    # Quick model pass for worker count
    if model is None:
        return "S1_normal"

    r = model.predict(
        img_bgr, device=device, verbose=False,
        conf=AUTO_QUICK_WORKER_CONF, iou=0.50, half=half,
        agnostic_nms=False,
    )[0]

    worker_cls_ids = {k for k, v in cls_map.items() if v in ("worker", "person")}
    ppe_cls_ids = {k for k, v in cls_map.items() if v in ("helmet", "safety_vest", "safety-vest")}

    worker_boxes = []
    ppe_boxes = []
    for b in (r.boxes or []):
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        if conf < AUTO_QUICK_WORKER_CONF:
            continue
        if cls_id in worker_cls_ids:
            worker_boxes.append(b.xyxy[0].cpu().numpy().tolist())
        elif cls_id in ppe_cls_ids:
            ppe_boxes.append(b.xyxy[0].cpu().numpy().tolist())

    n_workers = len(worker_boxes)
    img_h, img_w = img_bgr.shape[:2]
    frame_area = max(img_h * img_w, 1)

    if n_workers >= AUTO_CROWD_WORKER_THRESH:
        return "S4_crowded"

    if n_workers >= 3:
        # ── Crowd density score: workers per normalized 640×640 area ─────────
        # A density > 0.020 means ~8+ workers in a 640×640 equiv frame.
        density_score = n_workers / (frame_area / (640.0 * 640.0))
        if density_score >= 0.020:
            return "S4_crowded"

        # ── Overlap ratio: fraction of workers with ≥1 overlapping neighbour ─
        # Even with low mean IoU, if most workers touch each other → crowded.
        overlapping = set()
        for i in range(len(worker_boxes)):
            for j in range(i + 1, len(worker_boxes)):
                iou_val = iou_box(worker_boxes[i], worker_boxes[j])
                if iou_val >= AUTO_CROWD_OVERLAP_THRESH:
                    overlapping.add(i)
                    overlapping.add(j)
        overlap_ratio = len(overlapping) / n_workers
        if overlap_ratio >= 0.60:  # 60%+ of workers overlap a neighbour
            return "S4_crowded"

        # ── Mean pairwise IoU check (legacy signal, retained) ─────────────────
        all_ious = []
        for i in range(len(worker_boxes)):
            for j in range(i + 1, len(worker_boxes)):
                all_ious.append(iou_box(worker_boxes[i], worker_boxes[j]))
        if all_ious and float(np.mean(all_ious)) >= AUTO_CROWD_OVERLAP_THRESH:
            return "S4_crowded"

        # ── PPE Cluster gate: 5+ PPE inside the union of 3 workers → dense ───
        if len(ppe_boxes) >= 5:
            w_union = [
                min(b[0] for b in worker_boxes),
                min(b[1] for b in worker_boxes),
                max(b[2] for b in worker_boxes),
                max(b[3] for b in worker_boxes)
            ]
            inner_ppe = sum(1 for b in ppe_boxes if iou_box(b, w_union) > 0.1)
            if inner_ppe >= 5:
                return "S4_crowded"

    return "S1_normal"


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="BuildSight Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
database.init_db()

# Include GeoAI router
app.include_router(geoai_router, prefix="/api/geoai", tags=["geoai"])

# ── Turner AI Configuration ──────────────────────────────────────────────────
logger = logging.getLogger("buildsight")
turner_logger = logging.getLogger("buildsight.turner")

if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def log_turner_event(event: str, **payload):
    turner_logger.info(json.dumps({"event": event, **payload}, default=str))


def _safe_enum_name(value):
    if value is None:
        return None
    return getattr(value, "name", str(value))


def _extract_response_text(response):
    try:
        text = getattr(response, "text", None)
        if text:
            return text.strip()
    except Exception:
        pass

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        collected = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                collected.append(part_text)
        if collected:
            return "\n".join(collected).strip()
    return ""


def _response_block_metadata(response):
    prompt_feedback = getattr(response, "prompt_feedback", None)
    candidates = getattr(response, "candidates", None) or []

    block_reason = _safe_enum_name(getattr(prompt_feedback, "block_reason", None))
    candidate_finish_reasons = [
        _safe_enum_name(getattr(candidate, "finish_reason", None))
        for candidate in candidates
    ]

    safety_ratings = []
    for candidate in candidates:
        for rating in getattr(candidate, "safety_ratings", None) or []:
            safety_ratings.append({
                "category": _safe_enum_name(getattr(rating, "category", None)),
                "probability": _safe_enum_name(getattr(rating, "probability", None)),
                "blocked": getattr(rating, "blocked", None),
            })

    return {
        "prompt_block_reason": block_reason,
        "candidate_finish_reasons": candidate_finish_reasons,
        "safety_ratings": safety_ratings,
    }


def _turner_blocked_message(metadata: dict) -> str:
    categories = [
        rating["category"]
        for rating in metadata.get("safety_ratings", [])
        if rating.get("blocked") and rating.get("category")
    ]
    categories_text = ", ".join(categories) if categories else "safety policy"
    return (
        "I cannot provide a direct response because Gemini safety controls blocked this request. "
        f"Review the wording and retry with a narrower construction-safety question. Triggered category: {categories_text}."
    )


TURNER_SYSTEM_PROMPT = """
You are Turner, the Chief AI Site Supervisor for BuildSight.

Identity & Tone:
- Authoritative, disciplined, and strictly focused on site safety.
- Speak like a veteran construction foreperson: direct, concise, and action-oriented.
- Your goal is zero safety incidents.

Capabilities:
- You have a live data link to the site's telemetry (worker counts, PPE, environmental conditions, and alerts).
- You explain compliance gaps with high precision.
- You provide immediate, actionable safety interventions for site engineers.

Response Protocol:
1. Ground every response in the 'context' provided (live telemetry).
2. If compliance is suboptimal, call it out first and demand corrective action.
3. Keep responses compact. Use 2-5 authoritative bullet points.
4. If asked generic questions, answer through the lens of site safety and operator priority.
5. Do not hallucinate. If data is missing, state it plainly.
""".strip()

# ── Mistral AI (primary provider) ─────────────────────────────────────────────
MISTRAL_API_KEY   = os.environ.get("MISTRAL_API_KEY")
MISTRAL_MODEL     = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
MISTRAL_ENDPOINT  = "https://api.mistral.ai/v1/chat/completions"
mistral_enabled   = bool(MISTRAL_API_KEY)

if mistral_enabled:
    logger.info("Turner AI enabled. MISTRAL_API_KEY detected (model: %s).", MISTRAL_MODEL)
else:
    logger.warning("MISTRAL_API_KEY not found. Mistral AI disabled.")

# ── Google Gemini (fallback provider) ─────────────────────────────────────────
AI_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
ai_model = None

if AI_API_KEY and genai is not None:
    genai.configure(api_key=AI_API_KEY)
    ai_model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=TURNER_SYSTEM_PROMPT,
    )
    logger.info("Gemini fallback enabled. GOOGLE_API_KEY detected.")
elif AI_API_KEY and genai is None:
    logger.warning("GOOGLE_API_KEY present but google-generativeai is not installed. Gemini fallback disabled.")
else:
    logger.warning("GOOGLE_API_KEY not found. Gemini fallback disabled.")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    context: dict = {}


def _build_mistral_messages(req: "ChatRequest", full_prompt: str) -> list[dict]:
    """Convert ChatRequest history + enriched prompt into Mistral message list."""
    messages = [{"role": "system", "content": TURNER_SYSTEM_PROMPT}]
    for m in req.history:
        role = "user" if m.role == "user" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": full_prompt})
    return messages


def _call_mistral_sync(messages: list[dict]) -> str:
    """Synchronous Mistral chat call — run via run_in_threadpool."""
    resp = _requests.post(
        MISTRAL_ENDPOINT,
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": MISTRAL_MODEL, "messages": messages, "stream": False},
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _build_site_prompt(req: "ChatRequest") -> str:
    """Build the enriched site-context prompt from ChatRequest.context."""
    ctx       = req.context
    workers   = ctx.get("workers", 0)
    helmets   = ctx.get("helmets", 0)
    vests     = ctx.get("vests", 0)
    condition = ctx.get("condition", "Unknown")
    alerts    = ctx.get("alerts", [])
    telemetry = ctx.get("telemetry", {})
    return (
        f"LIVE SITE CONTEXT\n"
        f"- Active workers: {workers}\n"
        f"- Helmet compliance: {helmets}/{workers if workers else max(helmets, 1)}\n"
        f"- Vest compliance: {vests}/{workers if workers else max(vests, 1)}\n"
        f"- Site condition: {condition}\n"
        f"- Active escalations: {len(alerts)}\n"
        f"- Escalation details: {json.dumps(alerts, ensure_ascii=True)}\n"
        f"- Telemetry: {json.dumps(telemetry, ensure_ascii=True)}\n\n"
        f"USER REQUEST\n{req.message}"
    ).strip()


# ── model loading ──────────────────────────────────────────────────────────────
model_v11 = None
model_v26 = None
cls_map   = {}
mode_name = "unloaded"
gemini_auditor = None
gemini_auditor_enabled = False
gemini_auditor_model_name = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def init_gemini_auditor():
    global gemini_auditor, gemini_auditor_enabled, gemini_auditor_model_name

    gemini_auditor = None
    gemini_auditor_enabled = False
    gemini_auditor_model_name = None

    if not _env_flag("BUILDSIGHT_GEMINI_AUDITOR_ENABLED", default=True):
        logger.info("Gemini auditor disabled by BUILDSIGHT_GEMINI_AUDITOR_ENABLED.")
        return

    api_key = (
        os.environ.get("GEMINI_AUDITOR_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        logger.warning("Gemini auditor disabled. No API key configured.")
        return

    model_name = os.environ.get("GEMINI_AUDITOR_MODEL", "gemini-2.5-flash")

    try:
        from gemini_auditor import GeminiAuditor

        gemini_auditor = GeminiAuditor(api_key=api_key, model_name=model_name)
        gemini_auditor_enabled = True
        gemini_auditor_model_name = model_name
        logger.info("Gemini auditor enabled (model: %s).", model_name)
    except Exception as exc:
        logger.warning("Gemini auditor unavailable: %s", exc)

def load_models():
    global model_v11, model_v26, cls_map, mode_name
    from ultralytics import YOLO

    print(f"Device: {DEVICE}  |  FP16: {USE_HALF}")

    if SASTRA_V11.exists() and SASTRA_V26.exists():
        print("Loading SASTRA ensemble models...")
        model_v11 = YOLO(str(SASTRA_V11))
        model_v26 = YOLO(str(SASTRA_V26))
        cls_map   = SASTRA_CLASSES
        mode_name = "ensemble-wbf"
        print("Ensemble (YOLOv11 + YOLOv26) loaded.")
    elif SASTRA_V11.exists():
        model_v11 = YOLO(str(SASTRA_V11))
        cls_map   = SASTRA_CLASSES
        mode_name = "yolov11-single"
        print("YOLOv11 (single) loaded.")
    elif LOCAL_BEST.exists():
        print(f"Loading local base model: {LOCAL_BEST}")
        model_v11 = YOLO(str(LOCAL_BEST))
        cls_map   = LOCAL_CLASSES
        mode_name = "local-base"
        print("Local base model loaded.")
    else:
        print("ERROR: No model weights found.")

    # Warm up on GPU to JIT-compile kernels before first real request
    if model_v11 is not None and DEVICE.startswith("cuda"):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model_v11.predict(dummy, device=DEVICE, verbose=False,
                          conf=0.9, half=USE_HALF)
        if model_v26 is not None:
            model_v26.predict(dummy, device=DEVICE, verbose=False,
                              conf=0.9, half=USE_HALF)
        print("GPU warm-up done.")

load_models()
init_gemini_auditor()

# ── Two-layer ensemble post-processor (buildsight_ensemble.py) ────────────────
# Primary layer  : YOLOv11 + YOLOv26 WBF with geometric guardrails
#                  (from ensemble_inference.py.bak — 2026-04-10)
# Secondary layer: condition-aware WBF profiles + worker recovery + temporal PPE
#                  (from site_aware_ensemble.py.bak — 2026-04-11)
try:
    from buildsight_ensemble import (  # type: ignore[import]
        wbf_fuse           as _ensemble_wbf_primary,
        wbf_fuse_condition as _ensemble_wbf_postprocess,
        detect_condition   as _ensemble_detect_condition,
        preprocess_frame   as _ensemble_preprocess,
        TemporalPPEFilter,
    )
    _ensemble_temporal = TemporalPPEFilter(max_misses=6)
    _ENSEMBLE_POSTPROCESS_ENABLED = True
    logger.info("Two-layer ensemble post-processor loaded from buildsight_ensemble.py")
except Exception as _e:
    _ENSEMBLE_POSTPROCESS_ENABLED = False
    logger.warning("buildsight_ensemble.py unavailable (%s) — using legacy pipeline", _e)


# ── WBF (unchanged — exact same logic as ensemble_batch.py) ───────────────────
def iou_box(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(union, 1e-6)

def wbf_fuse(
    all_preds,
    img_w, img_h,
    iou_override:  dict | None = None,
    conf_gate:     dict | None = None,   # per-class post-WBF confidence gate
):
    iou_table  = iou_override if iou_override is not None else WBF_IOU
    gate_table = conf_gate    if conf_gate    is not None else POST_WBF_GLOBAL
    flat = []
    for midx, (boxes, scores, labels) in enumerate(all_preds):
        for box, score, label in zip(boxes, scores, labels):
            norm = [box[0]/img_w, box[1]/img_h, box[2]/img_w, box[3]/img_h]
            flat.append((norm, score, label, midx))
    if not flat:
        return []
    result = []
    for cls_id in range(3):
        cf = [(b, s, mi) for b, s, l, mi in flat if l == cls_id]
        if not cf:
            continue
        iou_t = iou_table.get(cls_id, 0.50)
        cf.sort(key=lambda x: -x[1])
        used = [False] * len(cf)
        for i in range(len(cf)):
            if used[i]:
                continue
            cb, cs, cm = [cf[i][0]], [cf[i][1]], [cf[i][2]]
            used[i] = True
            for j in range(i + 1, len(cf)):
                if not used[j] and iou_box(cf[i][0], cf[j][0]) >= iou_t:
                    cb.append(cf[j][0]); cs.append(cf[j][1]); cm.append(cf[j][2])
                    used[j] = True
            rw = np.array([MODEL_WEIGHTS[min(mi, len(MODEL_WEIGHTS)-1)] * s
                           for mi, s in zip(cm, cs)])
            nw = rw / rw.sum()
            fs = float(np.average(cs, weights=nw))
            if fs < gate_table.get(cls_id, 0.20):
                continue
            fb = np.average(cb, axis=0, weights=nw)
            # n_models: number of distinct model indices that contributed
            n_models = len(set(cm))
            result.append({
                "box":      [fb[0]*img_w, fb[1]*img_h, fb[2]*img_w, fb[3]*img_h],
                "score":    fs,
                "cls":      cls_id,
                "n_models": n_models,   # 1 = only one model agreed, 2 = both
            })
    return result


# ── Worker-to-PPE spatial association ─────────────────────────────────────────
# Minimum confidence a helmet must reach to be counted. Raises the bar so that
# low-confidence false positives on hair / shoulders / backs are ignored.
MIN_HELMET_CONF = THRESHOLD_STATE["helmet"]
GEMINI_AUDIT_MIN_CONF = THRESHOLD_STATE["vest"]
GEMINI_AUDIT_MAX_CONF = 0.65

# ── CLAHE instance (created once, reused per frame) ───────────────────────────
_clahe_cache: dict = {}

def _get_clahe(clip: float) -> cv2.CLAHE:
    """Return a cached CLAHE object for the given clip limit."""
    if clip not in _clahe_cache:
        _clahe_cache[clip] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return _clahe_cache[clip]

# Pre-computed gamma LUT cache (gamma → uint8 LUT)
_gamma_cache: dict = {}

def _gamma_lut(gamma: float) -> np.ndarray:
    if gamma not in _gamma_cache:
        inv = 1.0 / gamma
        _gamma_cache[gamma] = np.array(
            [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8
        )
    return _gamma_cache[gamma]


def enhance_frame(img_bgr: np.ndarray,
                  condition: str = "S1_normal",
                  clip: float = 2.0) -> np.ndarray:
    """
    Adaptive preprocessing pipeline per site condition.

    S1_normal:
      CLAHE on the L channel of LAB — boosts local contrast without colour
      shift, helping workers with cement-coloured clothing stand out.

    S2_dusty:
      CLAHE (clip 2.5) → bilateral filter (d=7, σ=90) → unsharp mask.
      Bilateral removes haze/grain while keeping worker silhouette edges
      sharp for YOLO. Unsharp mask recovers fine detail lost in dust haze.

    S3_low_light:
      CLAHE (clip 3.0) → gamma correction (γ=1.8) → slight denoise.
      Gamma lifts mid-tones so YOLO can extract features invisible in dark
      footage. Denoising removes ISO noise amplified by the gamma boost.

    S4_crowded:
      CLAHE (clip 2.0) → mild sharpening kernel.
      Sharpening improves separation of adjacent worker silhouettes so YOLO
      can resolve fine boundary gaps between closely-spaced people.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    l = _get_clahe(clip).apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)

    if condition == "S3_low_light":
        # Gamma 1.8 → lift dark mid-tones more aggressively than 1.5
        result = cv2.LUT(result, _gamma_lut(1.8))
        # Denoise ISO noise amplified by gamma (fast NL-means on rescaled img)
        result = cv2.fastNlMeansDenoisingColored(result, None, 6, 6, 7, 21)

    elif condition == "S2_dusty":
        # Bilateral: edge-preserving denoiser removes haze grain (σ raised 75→90)
        result = cv2.bilateralFilter(result, 7, 90, 90)
        # Unsharp mask to recover worker edge detail softened by the bilateral
        blur   = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.4, blur, -0.4, 0)

    elif condition == "S4_crowded":
        # Mild sharpening to improve edge separation between adjacent workers
        kernel = np.array([[ 0, -0.5,  0],
                           [-0.5,  3, -0.5],
                           [ 0, -0.5,  0]], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def is_valid_helmet(hbox: list, worker_bw: float, worker_bh: float) -> bool:
    """
    Geometric sanity check to reject false-positive helmets on hair, scarves,
    cloth, shoulders, or body regions:
      • Aspect ratio 0.5–2.0 (helmets are roughly round, not very elongated)
      • Width ≤ 70 % of worker box width (a head can't be wider than the torso)
      • Width ≥ 8 px and height ≥ 8 px (minimum size — not pixel noise)
      • Height ≤ 55 % of worker box height (head occupies upper portion only)
    """
    hx1, hy1, hx2, hy2 = hbox
    hw = max(hx2 - hx1, 1)
    hh = max(hy2 - hy1, 1)
    aspect = hw / hh
    if aspect > 2.2 or aspect < 0.45:   # too wide (scarf) or too tall (body)
        return False
    if hw > worker_bw * 0.70:            # wider than torso → not a head
        return False
    if hh > worker_bh * 0.55:            # taller than upper-half → body region
        return False
    if hw < 8 or hh < 8:                 # pixel noise
        return False
    return True


def is_valid_worker(
    wbox:        list,
    img_w:       int,
    img_h:       int,
    score:       float,
    model_count: int  = 1,     # how many models agreed on this box (1 or 2)
    condition:   str  = "S1_normal",
    nearby_ppe:  bool = False,  # True → helmet/vest found spatially adjacent
) -> bool:
    """
    Multi-signal validation rejecting cement bags, sand bags, brick stacks,
    tarpaulins, debris piles, scaffolding shadows, pipes, and machinery parts
    misclassified as workers.

    ── PPE fast-bypass ───────────────────────────────────────────────────────
    If a helmet or vest is spatially near the candidate worker box (computed
    before this call in run_inference) we skip all geometry checks — confirmed
    PPE proves it's a person regardless of box shape.

    ── Signal 1: Landscape aspect reject ────────────────────────────────────
    Workers in all real-world poses are taller than wide from CCTV angles.
    Cement bags, brick stacks, and bundled pipes are landscape-shaped.

    ── Signal 2: Size floor ─────────────────────────────────────────────────
    Real workers at any practical CCTV distance occupy ≥ 1.0-1.2% of the
    shorter image dimension in both width and height.

    ── Signal 3: Single-model agreement gate ────────────────────────────────
    One-model-only detections require higher confidence. Cement bags with shape
    similarity are typically caught by only one model at 0.25–0.45.

    ── Signal 4: Hard-landscape reject (aspect > 3.2) ───────────────────────
    Pipes, long brick stacks, and tarpaulin ridges — never a person.

    ── Signal 5: Portrait check (model-count independent) ───────────────────
    Even when both models agree, nearly-square/landscape boxes need higher
    confidence. Cement bags on plank stacks produce ~1:1 boxes that fool both
    models. Thresholds are per-condition to avoid rejecting crouching workers
    in dusty/low-light scenes where boxes expand due to haze.

    ── Signal 6: Box-width ceiling ──────────────────────────────────────────
    No single worker spans > 45% of frame width. Wider = material pile.

    ── Signal 7: Height floor ───────────────────────────────────────────────
    Workers must occupy ≥ 6–8% of image height. Below this = pixel noise or
    distant objects that look like construction materials at low resolution.

    ── Signal 8: Area ceiling ───────────────────────────────────────────────
    A single person cannot cover > 18% of total image area. Oversized blobs
    wrap multi-object regions, not a single standing worker.
    """
    # ── PPE fast-bypass ────────────────────────────────────────────────────
    if nearby_ppe:
        return True

    x1, y1, x2, y2 = wbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    aspect = bw / bh   # > 1.0 = landscape; < 1.0 = portrait (person = portrait)

    # ── Signal 4: hard-landscape reject ───────────────────────────────────
    if aspect > 3.2:
        return False

    # ── Signal 6: box-width ceiling ───────────────────────────────────────
    if bw > img_w * 0.45:
        return False

    # ── Signal 8: area ceiling ─────────────────────────────────────────────
    if (bw * bh) > (img_w * img_h * 0.18):
        return False

    # ── Signal 7: height floor ─────────────────────────────────────────────
    # S4_crowded: 0.045→0.025 — workers standing on elevated walls/platforms
    #   seen from a below-camera angle have much shorter bounding boxes than
    #   ground-level workers at the same distance. At 640px frame height,
    #   0.025 = 16px minimum — still safely above pure-noise blobs.
    # S1_normal: 0.08→0.07 — slight relaxation to catch distant workers at
    #   the far edge of frame who are genuinely small but clearly human.
    if condition == "S4_crowded":
        min_h_frac = 0.025
    elif condition in ("S2_dusty", "S3_low_light"):
        min_h_frac = 0.06
    else:
        min_h_frac = 0.07   # S1_normal — was 0.08, relaxed for edge workers
    if bh < img_h * min_h_frac and score < 0.60:
        return False

    # ── Signal 1: aspect ratio per condition ───────────────────────────────
    # S4_crowded: allow wider boxes (adjacent workers create merged-look boxes)
    # S2/S3:      haze/noise inflates bounding boxes beyond person outline
    # S1_normal:  strict portrait — clear sky background, no interference
    if condition == "S4_crowded":
        max_aspect = 2.4
    elif condition in ("S3_low_light", "S2_dusty"):
        max_aspect = 2.2
    else:
        max_aspect = 2.0   # S1_normal: tight portrait requirement

    if aspect > max_aspect and score < 0.68:
        return False

    # ── Signal 2: size floor ───────────────────────────────────────────────
    # Reduced for S4_crowded to catch small distant workers.
    size_frac = 0.008 if condition == "S4_crowded" else (0.010 if condition == "S3_low_light" else 0.011)
    min_px    = min(img_w, img_h) * size_frac
    if bw < min_px or bh < min_px:
        return False

    # ── Signal 3: single-model agreement gate ─────────────────────────────
    if model_count == 1:
        if condition == "S3_low_light":
            single_thresh = 0.32
        elif condition == "S2_dusty":
            single_thresh = 0.36
        elif condition == "S4_crowded":
            # RECALL FIX: 0.40→0.28 — elevated/distant/occluded workers that only
            # one model detects typically score 0.25–0.38. The other model misses
            # them due to partial occlusion or extreme viewing angle. Clutter FPs
            # are still caught by aspect ratio, height floor, and PPE-support gates.
            single_thresh = 0.28
        else:
            single_thresh = 0.50   # S1_normal: cement bags rarely score > 0.50 solo
        if score < single_thresh:
            return False

    # ── Signal 5: portrait check (applies regardless of model_count) ───────
    # Calibrated per-condition — catches boxes that fool BOTH models:
    #   S1_normal:   aspect > 0.88 and score < 0.70 → reject (bags on planks)
    #   S2_dusty:    aspect > 0.95 and score < 0.65 → reject (haze inflates boxes)
    #   S3_low_light: aspect > 1.00 and score < 0.62 → reject (dark scene noise)
    #   S4_crowded:  tightened — overlapping workers still produce portrait-ish
    #                boxes; wooden poles, rebar, and horizontal pipes are the
    #                main landscape FPs at low confidence in crowded scenes.
    if condition == "S1_normal":
        if aspect > 0.88 and score < 0.70:
            return False
    elif condition == "S2_dusty":
        if aspect > 0.95 and score < 0.65:
            return False
    elif condition == "S3_low_light":
        if aspect > 1.00 and score < 0.62:
            return False
    elif condition == "S4_crowded":
        # RECALL FIX: widened portrait check 1.60→2.00, score gate 0.62→0.55.
        # Workers viewed from below on elevated walls have wider apparent aspect
        # ratios (up to ~1.8) because the camera foreshortens their height.
        # Genuine landscape FPs (planks, rebar, scaffolding) typically land at
        # aspect > 2.0 and can be caught at the lower score gate of 0.55.
        if aspect > 2.00 and score < 0.55:
            return False

    return True


def associate_ppe_to_workers(raw: list) -> list:
    """
    Link helmet and vest detections to their parent worker after YOLO/WBF.

    Search zones (expanded from the strict containment of v1):
      • Helmet: upper 75 % of worker height, ±15 % horizontal padding,
        10 % vertical headroom above the worker box (helmet often sits above
        the box top in side/rear-view poses). Minimum conf = MIN_HELMET_CONF
        to reject false positives on hair, shoulders, and backs.
      • Vest: full worker height + 20 % below (vest commonly extends below a
        tight worker box), ±15 % horizontal padding.

    Matching priority: closest Euclidean distance from the PPE centre to the
    worker centre (rather than horizontal-only distance), so partially-occluded
    PPE items on the correct worker win over nearby items on adjacent workers.
    One-to-one assignment — once a PPE item is claimed it cannot be re-used.
    """
    worker_ids = {k for k, v in cls_map.items() if v in ("worker", "person")}
    helmet_ids = {k for k, v in cls_map.items() if v == "helmet"}
    vest_ids   = {k for k, v in cls_map.items()
                  if v in ("safety_vest", "safety-vest")}

    workers = [(i, d) for i, d in enumerate(raw) if d["cls"] in worker_ids]
    helmets = [(i, d) for i, d in enumerate(raw) if d["cls"] in helmet_ids]
    vests   = [(i, d) for i, d in enumerate(raw) if d["cls"] in vest_ids]

    matched_helmets: set = set()
    matched_vests:   set = set()
    result = [dict(d) for d in raw]

    for wi, wd in workers:
        bx1, by1, bx2, by2 = wd["box"]
        bw  = bx2 - bx1
        bh  = by2 - by1
        wcx = (bx1 + bx2) / 2
        wcy = (by1 + by2) / 2

        # Expanded search rectangles
        h_pad      = bw * 0.15          # horizontal margin
        h_head     = bh * 0.10          # allow helmet slightly above worker box

        helm_x1 = bx1 - h_pad
        helm_x2 = bx2 + h_pad
        helm_y1 = by1 - h_head          # 10 % headroom above box
        helm_y2 = by1 + bh * 0.75      # upper 75 % of worker height

        vest_x1 = bx1 - h_pad
        vest_x2 = bx2 + h_pad
        vest_y1 = by1
        vest_y2 = by2 + bh * 0.20      # 20 % below box bottom

        # ── Helmet matching ──────────────────────────────────────────────────
        best_hi, best_h_dist = -1, float("inf")
        for hi, hd in helmets:
            if hi in matched_helmets:
                continue
            # Reject weak detections (hair, shoulder false positives)
            if hd.get("score", 1.0) < THRESHOLD_STATE["helmet"]:
                continue
            # Geometric sanity: shape/size must match a real helmet
            if not is_valid_helmet(hd["box"], bw, bh):
                continue
            hx1, hy1, hx2, hy2 = hd["box"]
            hcx = (hx1 + hx2) / 2
            hcy = (hy1 + hy2) / 2
            if helm_x1 <= hcx <= helm_x2 and helm_y1 <= hcy <= helm_y2:
                dist = ((hcx - wcx) ** 2 + (hcy - wcy) ** 2) ** 0.5
                if dist < best_h_dist:
                    best_hi, best_h_dist = hi, dist
        has_helmet = best_hi >= 0
        if has_helmet:
            matched_helmets.add(best_hi)

        # ── Vest matching ────────────────────────────────────────────────────
        best_vi, best_v_dist = -1, float("inf")
        for vi, vd in vests:
            if vi in matched_vests:
                continue
            vx1, vy1, vx2, vy2 = vd["box"]
            vcx = (vx1 + vx2) / 2
            vcy = (vy1 + vy2) / 2
            if vest_x1 <= vcx <= vest_x2 and vest_y1 <= vcy <= vest_y2:
                dist = ((vcx - wcx) ** 2 + (vcy - wcy) ** 2) ** 0.5
                if dist < best_v_dist:
                    best_vi, best_v_dist = vi, dist
        has_vest = best_vi >= 0
        if has_vest:
            matched_vests.add(best_vi)

        result[wi]["has_helmet"] = has_helmet
        result[wi]["has_vest"]   = has_vest

    # Remove matched PPE boxes — they are now represented by the flags on their
    # parent worker.  Unmatched PPE (no nearby worker) is kept so the canvas
    # still draws loose helmet/vest items that the tracker missed as a worker.
    for mi in matched_helmets:
        result[mi]["_matched"] = True
    for mi in matched_vests:
        result[mi]["_matched"] = True
    return [d for d in result if not d.get("_matched", False)]


# ── Class-ID helpers (module-level, computed once after load_models) ──────────
def _worker_ids() -> set:
    return {k for k, v in cls_map.items() if v in ("worker", "person")}

def _helmet_ids() -> set:
    return {k for k, v in cls_map.items() if v == "helmet"}

def _vest_ids() -> set:
    return {k for k, v in cls_map.items() if v in ("safety_vest", "safety-vest")}


def _class_id_for_name(class_name: str) -> int | None:
    normalized = {
        "person": "worker",
        "safety-vest": "safety_vest",
    }.get(class_name, class_name)
    for cls_id, mapped_name in cls_map.items():
        candidate = "safety_vest" if mapped_name == "safety-vest" else mapped_name
        if candidate == normalized:
            return cls_id
    return None


def _auditor_condition_name(condition: str) -> str:
    mapping = {
        "S1_normal": "normal",
        "S2_dusty": "dusty",
        "S3_low_light": "low_light",
        "S4_crowded": "crowded",
    }
    return mapping.get(condition, condition or "normal")


def _to_auditor_detection(det: dict) -> dict | None:
    class_name = cls_map.get(det["cls"], "unknown")
    normalized_name = {
        "person": "worker",
        "safety-vest": "safety_vest",
    }.get(class_name, class_name)
    if normalized_name not in {"worker", "helmet", "safety_vest"}:
        return None
    auditor_class_id = {"helmet": 0, "safety_vest": 1, "worker": 2}[normalized_name]
    return {
        "class_id": auditor_class_id,
        "class_name": normalized_name,
        "xyxy": [float(v) for v in det["box"]],
        "score": float(det["score"]),
    }


def _from_auditor_detection(det: dict) -> dict | None:
    class_name = det.get("class_name")
    cls_id = _class_id_for_name(class_name)
    if cls_id is None:
        return None
    return {
        "box": [float(v) for v in det["xyxy"]],
        "score": float(det.get("score", 0.0)),
        "cls": cls_id,
        "n_models": 0 if det.get("source") == "gemini" else 1,
        "source": det.get("source", "gemini_auditor"),
    }


def _looks_suspicious_for_gemini(
    det: dict,
    all_detections: list,
    img_w: int,
    img_h: int,
) -> bool:
    score = float(det.get("score", 0.0))
    if score < THRESHOLD_STATE["vest"] or score > GEMINI_AUDIT_MAX_CONF:
        return False

    cls_name = cls_map.get(det["cls"], "")
    box = det["box"]
    bw = max(box[2] - box[0], 1.0)
    bh = max(box[3] - box[1], 1.0)
    worker_boxes = [d for d in all_detections if d["cls"] in _worker_ids()]

    if cls_name in ("worker", "person"):
        return (not _has_nearby_ppe(box, all_detections)) and int(det.get("n_models", 1)) == 1

    if cls_name == "helmet":
        matched = False
        for worker in worker_boxes:
            wx1, wy1, wx2, wy2 = worker["box"]
            wcx = (wx1 + wx2) / 2.0
            hcx = (box[0] + box[2]) / 2.0
            hcy = (box[1] + box[3]) / 2.0
            pad = (wx2 - wx1) * 0.15
            if wx1 - pad <= hcx <= wx2 + pad and wy1 - (wy2 - wy1) * 0.10 <= hcy <= wy1 + (wy2 - wy1) * 0.75:
                matched = True
                break
        return (not matched) or (not is_valid_helmet(box, bw * 1.6, bh * 2.4))

    if cls_name in ("safety_vest", "safety-vest"):
        matched = False
        for worker in worker_boxes:
            wx1, wy1, wx2, wy2 = worker["box"]
            vcx = (box[0] + box[2]) / 2.0
            vcy = (box[1] + box[3]) / 2.0
            pad = (wx2 - wx1) * 0.15
            if wx1 - pad <= vcx <= wx2 + pad and wy1 <= vcy <= wy2 + (wy2 - wy1) * 0.20:
                matched = True
                break
        return (not matched) or bw > img_w * 0.18 or bh > img_h * 0.22

    return False


def apply_gemini_audit(img_bgr: np.ndarray, detections: list, condition: str, enabled: bool) -> tuple[list, dict]:
    meta = {
        "requested": bool(enabled),
        "applied": False,
        "available": gemini_auditor_enabled,
        "model": gemini_auditor_model_name,
        "mode": "validator_only",
        "kept": len(detections),
        "rejected": 0,
        "added": 0,
        "ignored_supplements": 0,
        "reviewed_candidates": 0,
    }

    if not enabled or not gemini_auditor_enabled or gemini_auditor is None or not detections:
        return detections, meta

    img_h, img_w = img_bgr.shape[:2]
    auditor_input = []
    source_indices = []
    for idx, det in enumerate(detections):
        if not _looks_suspicious_for_gemini(det, detections, img_w, img_h):
            continue
        converted = _to_auditor_detection(det)
        if converted is None:
            continue
        auditor_input.append(converted)
        source_indices.append(idx)

    if not auditor_input:
        return detections, meta
    meta["reviewed_candidates"] = len(auditor_input)

    try:
        new_dets, kept_dets = gemini_auditor.audit(
            img_bgr,
            auditor_input,
            _auditor_condition_name(condition),
        )
    except Exception as exc:
        logger.warning("Gemini auditor request failed: %s", exc)
        meta["error"] = str(exc)
        return detections, meta

    kept_counts: dict[str, int] = {}
    for det in kept_dets:
        key = json.dumps(det, sort_keys=True)
        kept_counts[key] = kept_counts.get(key, 0) + 1

    rejected_positions = set()
    for source_idx, det in zip(source_indices, auditor_input):
        key = json.dumps(det, sort_keys=True)
        if kept_counts.get(key, 0) > 0:
            kept_counts[key] -= 1
        else:
            rejected_positions.add(source_idx)
    merged = [det for idx, det in enumerate(detections) if idx not in rejected_positions]

    meta["applied"] = True
    meta["kept"] = len(merged)
    meta["rejected"] = len(rejected_positions)
    meta["added"] = 0
    meta["ignored_supplements"] = len(new_dets)
    return merged, meta


def _has_nearby_ppe(wbox: list, raw: list) -> bool:
    """
    Return True if any helmet or vest detection centre falls within the
    expanded PPE search zone around `wbox`.  Mirrors the zones used by
    associate_ppe_to_workers so the pre-filter and the association step
    see the same geometry.
    """
    hids = _helmet_ids()
    vids = _vest_ids()
    bx1, by1, bx2, by2 = wbox
    bw  = bx2 - bx1
    bh  = by2 - by1
    pad = bw * 0.15
    # Helmet zone: upper 75% + 10% headroom above
    hx1, hx2 = bx1 - pad, bx2 + pad
    hy1, hy2 = by1 - bh * 0.10, by1 + bh * 0.75
    # Vest zone: full height + 20% below
    vx1, vx2 = bx1 - pad, bx2 + pad
    vy1, vy2 = by1, by2 + bh * 0.20
    for d in raw:
        if d["cls"] not in hids and d["cls"] not in vids:
            continue
        cx = (d["box"][0] + d["box"][2]) / 2
        cy = (d["box"][1] + d["box"][3]) / 2
        if d["cls"] in hids:
            if hx1 <= cx <= hx2 and hy1 <= cy <= hy2:
                return True
        else:
            if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                return True
    return False


# ── Tile-based inference for crowded/distant worker recall ────────────────────
def _infer_tiles(img_bgr: np.ndarray, model_v11, model_v26, device: str,
                 half: bool, pre_conf: float, nms_iou: float,
                 wbf_iou_cond: dict, cond_conf_gate: dict) -> list:
    """
    Divide the frame into a 2×2 grid with 20% overlap, run inference on each
    tile, and merge results back into full-frame pixel coordinates.

    This recovers workers near edges and corners that full-frame inference
    misses when they are small, partially occluded, or elevated.

    Returns merged raw detections (same format as wbf_fuse output).
    """
    h, w = img_bgr.shape[:2]
    tile_w = int(w * 0.60)  # 60% of frame width → 20% overlap in the centre
    tile_h = int(h * 0.60)
    tile_origins = [
        (0,           0          ),   # top-left
        (w - tile_w,  0          ),   # top-right
        (0,           h - tile_h ),   # bottom-left
        (w - tile_w,  h - tile_h ),   # bottom-right
    ]

    all_results: list[dict] = []

    for ox, oy in tile_origins:
        tile = img_bgr[oy:oy + tile_h, ox:ox + tile_w]
        tw, th = tile.shape[1], tile.shape[0]

        r11 = model_v11.predict(
            tile, device=device, verbose=False,
            conf=pre_conf, iou=nms_iou, half=half, agnostic_nms=False,
        )[0]
        boxes11 = [b.xyxy[0].cpu().numpy().tolist() for b in (r11.boxes or [])]
        scores11 = [float(b.conf[0]) for b in (r11.boxes or [])]
        labels11 = [int(b.cls[0]) for b in (r11.boxes or [])]

        if model_v26 is not None:
            r26 = model_v26.predict(
                tile, device=device, verbose=False,
                conf=pre_conf, iou=nms_iou, half=half, agnostic_nms=False,
            )[0]
            boxes26 = [b.xyxy[0].cpu().numpy().tolist() for b in (r26.boxes or [])]
            scores26 = [float(b.conf[0]) for b in (r26.boxes or [])]
            labels26 = [int(b.cls[0]) for b in (r26.boxes or [])]
            tile_dets = wbf_fuse(
                [(boxes11, scores11, labels11), (boxes26, scores26, labels26)],
                tw, th, iou_override=wbf_iou_cond, conf_gate=cond_conf_gate,
            )
        else:
            tile_dets = [
                {"box": b, "score": s, "cls": l, "n_models": 1}
                for b, s, l in zip(boxes11, scores11, labels11)
            ]

        # Remap tile coordinates back to full-frame space
        for det in tile_dets:
            b = det["box"]
            det["box"] = [b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy]
            all_results.append(det)

    return all_results


def _merge_full_and_tile(full_dets: list, tile_dets: list) -> list:
    """
    Merge full-frame detections with tile detections.
    Tiles may find workers the full-frame pass missed (small, edge, elevated).
    Deduplicates by IoU ≥ 0.35 within the same class, keeping the higher score.
    """
    if not tile_dets:
        return full_dets
    if not full_dets:
        return tile_dets

    final = list(full_dets)
    for td in tile_dets:
        duplicate = False
        for fd in final:
            if fd["cls"] != td["cls"]:
                continue
            # Lowered 0.45->0.32: exact duplicate tile/full-frame merge cleanup
            if iou_box(td["box"], fd["box"]) >= 0.32:
                duplicate = True
                if td["score"] > fd["score"]:
                    fd["score"] = td["score"]
                break
        if not duplicate:
            final.append(td)
    return final


def _final_worker_nms(raw: list, iou_thresh: float = 0.35) -> list:
    """
    Final intra-class NMS applied to worker boxes AFTER all merging stages.
    Threshold lowered 0.45 -> 0.35 per user preference for aggressive duplicate cleanup.
    """
    worker_ids = _worker_ids()
    workers = sorted(
        [d for d in raw if d["cls"] in worker_ids],
        key=lambda x: -x["score"],
    )
    others = [d for d in raw if d["cls"] not in worker_ids]

    kept: list[dict] = []
    for w in workers:
        overlap = any(iou_box(w["box"], k["box"]) >= iou_thresh for k in kept)
        if not overlap:
            kept.append(w)

    return others + kept


# ── Gemini auditor rate-limiter ───────────────────────────────────────────────
# Prevents Gemini from becoming an inference bottleneck in live-video mode.
# At most GEMINI_RATE_LIMIT_PER_N frames will trigger an audit call; between
# rate-limit frames the audit is skipped and detections pass through untouched.
GEMINI_RATE_LIMIT_PER_N = 10   # one Gemini call at most every 10 inference frames
_gemini_frame_counter: int = 0


def _should_run_gemini() -> bool:
    global _gemini_frame_counter
    _gemini_frame_counter += 1
    return (_gemini_frame_counter % GEMINI_RATE_LIMIT_PER_N) == 1


# ── inference core (runs in threadpool — never blocks the event loop) ─────────
def run_inference(
    img_bgr:     np.ndarray,
    condition:   str   = "S1_normal",
    conf_thresh: float = PRE_CONF,
    class_conf:  dict | None = None,
    nms_iou_map: dict | None = None,
    wbf_iou_map: dict | None = None,
    clahe:       bool  = True,
    clahe_clip:  float = 2.0,
    gemini_audit: bool = False,
    use_tiling:  bool  = False,
):
    audit_img_bgr = img_bgr.copy()

    # ── Auto scene classification ─────────────────────────────────────────────
    # Image-stats checks run every frame (~1ms). The expensive model forward-pass
    # for S4 worker-count detection only runs every SCENE_CACHE_FRAMES frames.
    if condition == "auto":
        gray_quick = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness_quick = float(np.mean(gray_quick))
        if brightness_quick < AUTO_LOW_LIGHT_THRESH:
            raw_cond = "S3_low_light"
        else:
            hsv_quick = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            contrast_quick = float(np.std(gray_quick))
            sat_quick = float(np.mean(hsv_quick[:, :, 1]))
            if contrast_quick < AUTO_DUSTY_STD_THRESH and sat_quick < AUTO_DUSTY_SAT_THRESH:
                raw_cond = "S2_dusty"
            elif _scene_tracker.should_reclassify():
                # Full model pass — only every SCENE_CACHE_FRAMES frames
                raw_cond = classify_scene_fast(img_bgr, model_v11, DEVICE, USE_HALF)
                _scene_tracker._last_raw_cond = raw_cond
            else:
                # Reuse last raw reading — no GPU call this frame
                raw_cond = _scene_tracker._last_raw_cond
        condition = _scene_tracker.update(raw_cond)

    if clahe:
        effective_clip = clahe_clip
        if condition == "S3_low_light" and clahe_clip < 3.0:
            effective_clip = 3.0
        elif condition == "S2_dusty" and clahe_clip < 2.5:
            effective_clip = 2.5
        img_bgr = enhance_frame(img_bgr, condition, effective_clip)

    h, w = img_bgr.shape[:2]

    if condition == "S4_crowded":
        pre_conf = min(conf_thresh, 0.15)
    elif condition == "S3_low_light":
        pre_conf = min(conf_thresh, 0.15)
    elif condition == "S2_dusty":
        pre_conf = min(conf_thresh, 0.17)
    else:
        pre_conf = conf_thresh

    if condition == "S4_crowded":
        # Tuned per user preference: 0.45-0.50 for crowded-scene WBF fusion
        wbf_iou_cond = {0: 0.45, 1: 0.45, 2: 0.48}
        base_nms_iou = 0.65
    elif condition == "S3_low_light":
        wbf_iou_cond = {0: 0.40, 1: 0.45, 2: 0.60}
        base_nms_iou = NMS_IOU
    else:
        wbf_iou_cond = dict(WBF_IOU)
        base_nms_iou = NMS_IOU

    cond_conf_gate = POST_WBF_BY_CONDITION.get(condition, POST_WBF_GLOBAL)

    def _cls_idx(name: str) -> int:
        for k, v in cls_map.items():
            if v == name or (name == "vest" and v in ("safety_vest", "safety-vest")):
                return k
        return -1

    if wbf_iou_map:
        for name, val in wbf_iou_map.items():
            idx = _cls_idx(name)
            if idx >= 0:
                wbf_iou_cond[idx] = float(val)

    nms_iou = max(float(v) for v in nms_iou_map.values()) if nms_iou_map else base_nms_iou

    r11 = model_v11.predict(
        img_bgr, device=DEVICE, verbose=False,
        conf=pre_conf, iou=nms_iou, half=USE_HALF,
        agnostic_nms=False,
    )[0]
    boxes11, scores11, labels11 = [], [], []
    for box in (r11.boxes or []):
        boxes11.append(box.xyxy[0].cpu().numpy().tolist())
        scores11.append(float(box.conf[0]))
        labels11.append(int(box.cls[0]))

    if model_v26 is not None:
        force_ensemble = (condition == "S4_crowded")
        early_exit = (
            not force_ensemble and len(scores11) >= 1 and
            all(s >= max(EARLY_EXIT_CONF, pre_conf + 0.1) for s in scores11)
        )
        if early_exit:
            raw = [{"box": b, "score": s, "cls": l, "n_models": 1}
                   for b, s, l in zip(boxes11, scores11, labels11)]
            used_ensemble = False
        else:
            r26 = model_v26.predict(
                img_bgr, device=DEVICE, verbose=False,
                conf=pre_conf, iou=nms_iou, half=USE_HALF,
                agnostic_nms=False,
            )[0]
            boxes26 = [b.xyxy[0].cpu().numpy().tolist() for b in (r26.boxes or [])]
            scores26 = [float(b.conf[0]) for b in (r26.boxes or [])]
            labels26 = [int(b.cls[0]) for b in (r26.boxes or [])]

            all_preds = [(boxes11, scores11, labels11), (boxes26, scores26, labels26)]

            if _ENSEMBLE_POSTPROCESS_ENABLED:
                # ── PRIMARY LAYER (ensemble_inference.py.bak style) ──────────
                # WBF with geometric guardrails: landscape-box rejection +
                # area check embedded directly in the fusion step.
                raw = _ensemble_wbf_primary(all_preds, w, h)

                # ── POST-PROCESSING LAYER (site_aware_ensemble.py.bak style) ─
                # Condition-tuned second-pass WBF with per-scene IoU profiles,
                # tighter post-gates, and S4 crowded-worker recovery.
                raw = _ensemble_wbf_postprocess(raw, all_preds, w, h, condition)

                logger.debug("[ENSEMBLE] primary→%d dets after two-layer fusion (condition=%s)",
                             len(raw), condition)
            else:
                # Fallback: legacy single-pass WBF
                raw = wbf_fuse(
                    all_preds, w, h,
                    iou_override=wbf_iou_cond, conf_gate=cond_conf_gate,
                )

            used_ensemble = True
    else:
        raw = [{"box": b, "score": s, "cls": l, "n_models": 1}
               for b, s, l in zip(boxes11, scores11, labels11)]
        used_ensemble = False

    # RECALL FIX: For S4_crowded always run tile inference — elevated workers
    # on walls and workers near top/side edges are consistently missed by the
    # full-frame pass even when the scene already has 4+ detected workers.
    # For other modes keep the "< 3 workers" trigger to save GPU time.
    if use_tiling or condition == "S4_crowded":
        full_worker_count = sum(1 for d in raw if d["cls"] in _worker_ids())
        run_tiles = use_tiling or condition == "S4_crowded" or full_worker_count < 3
        if run_tiles:
            tile_dets = _infer_tiles(
                img_bgr, model_v11, model_v26, DEVICE, USE_HALF,
                pre_conf, nms_iou, wbf_iou_cond, cond_conf_gate,
            )
            raw = _merge_full_and_tile(raw, tile_dets)
            used_ensemble = True

    worker_cls_ids = _worker_ids()
    raw = [
        d for d in raw
        if d["cls"] not in worker_cls_ids
        or is_valid_worker(
            d["box"], w, h, d["score"],
            model_count=d.get("n_models", 1),
            condition=condition,
            nearby_ppe=_has_nearby_ppe(d["box"], raw),
        )
    ]

    _gemini_enabled = gemini_audit and _should_run_gemini()
    raw, audit_meta = apply_gemini_audit(audit_img_bgr, raw, condition, _gemini_enabled)
    raw = associate_ppe_to_workers(raw)
    raw = _final_worker_nms(raw, iou_thresh=0.35)

    effective_thresholds = dict(THRESHOLD_STATE)
    if class_conf:
        effective_thresholds["worker"] = max(float(class_conf.get("worker", THRESHOLD_STATE["worker"])), THRESHOLD_STATE["global_floor"])
        effective_thresholds["helmet"] = max(float(class_conf.get("helmet", THRESHOLD_STATE["helmet"])), THRESHOLD_STATE["global_floor"])
        effective_thresholds["vest"] = max(float(class_conf.get("vest", THRESHOLD_STATE["vest"])), THRESHOLD_STATE["global_floor"])

    # ── S4 recall override: lower final threshold gate for crowded scenes ─────
    # The model-trained thresholds in THRESHOLD_STATE are calibrated for S1.
    # S4 elevated/distant workers score 0.35–0.45 after two-model WBF fusion.
    # Single-model or scaffolding FPs typically score 0.28–0.34, so 0.34 cuts
    # those while retaining real workers whose fused score lands at 0.35+.
    if condition == "S4_crowded":
        effective_thresholds["worker"] = min(effective_thresholds["worker"], 0.34)
        effective_thresholds["helmet"] = min(effective_thresholds["helmet"], 0.22)
        effective_thresholds["vest"]   = min(effective_thresholds["vest"],   0.22)

    worker_detections = []
    helmet_detections = []
    vest_detections = []
    for det in raw:
        cls_name = cls_map.get(det["cls"], "")
        if cls_name in ("worker", "person") and det["score"] >= effective_thresholds["worker"]:
            worker_detections.append(dict(det, confidence=det["score"]))
        elif cls_name == "helmet" and det["score"] >= effective_thresholds["helmet"]:
            helmet_detections.append(det)
        elif cls_name in ("safety_vest", "safety-vest") and det["score"] >= effective_thresholds["vest"]:
            vest_detections.append(det)

    global _inference_frame_count
    _inference_frame_count += 1

    logger.debug("[STAGE 1] after threshold filter: %d workers  %d helmets  %d vests",
               len(worker_detections), len(helmet_detections), len(vest_detections))

    # ── S4 recall: temporarily relax height and human-score floors ───────────
    # ValidWorkerValidator and MaterialSuppressionLayer use global thresholds
    # calibrated for S1. For S4 elevated/distant workers we lower per-frame:
    #   min_height_px: 40→14px — wall workers appear small from below-camera
    #   min_human_score: 0.35→0.24 — distant workers have lower h/w ratios
    # Multi-gate suppression (aspect, motion, PPE, clutter mask) still catches
    # non-human blobs that pass the relaxed size/score gates.
    _prev_validator_h    = worker_validator.min_height_px
    _prev_suppressor_h   = material_suppressor.min_worker_height_px
    _prev_validator_hsco = worker_validator.min_human_score
    if condition == "S4_crowded":
        worker_validator.min_height_px    = 14
        worker_validator.min_human_score  = 0.24
        material_suppressor.min_worker_height_px = 14

    annotated_workers = worker_track_state.annotate(worker_detections)
    frame_area = h * w
    surviving_workers = []
    suppressed_count = 0
    for det in annotated_workers:
        has_ppe = compute_has_ppe(det, helmet_detections, vest_detections)
        det["has_ppe_nearby"] = has_ppe
        if material_suppressor.should_suppress(det, img_bgr, frame_area, det.get("is_static", False), has_ppe):
            suppressed_count += 1
            det["suppressed"] = True
        else:
            surviving_workers.append(det)

    logger.debug("[STAGE 2] after material suppression: %d surviving  %d suppressed",
               len(surviving_workers), suppressed_count)

    valid_workers, valid_worker_count = worker_validator.get_valid_workers(
        surviving_workers, effective_thresholds["worker"]
    )

    logger.debug("[STAGE 3] after valid-worker validation: %d valid", valid_worker_count)

    # Periodic state cleanup every 150 frames (~13s at 11fps) to prevent unbounded
    # accumulation of stale track counters as old worker IDs are recycled.
    # Restore global floors after S4-scoped relaxation
    worker_validator.min_height_px   = _prev_validator_h
    worker_validator.min_human_score = _prev_validator_hsco
    material_suppressor.min_worker_height_px = _prev_suppressor_h

    if _inference_frame_count % 150 == 0:
        active_ids = {d.get("track_id") for d in valid_workers if d.get("track_id") is not None}
        material_suppressor.cleanup_stale_counters(active_ids)
        worker_validator.cleanup_stale_tracks(active_ids)

    canonical_worker_cls = next(iter(worker_cls_ids)) if worker_cls_ids else 2
    for det in valid_workers:
        det["cls"] = canonical_worker_cls

    final_detections = valid_workers + helmet_detections + vest_detections
    final_detections = associate_ppe_to_workers(final_detections)
    valid_workers = [d for d in final_detections if d["cls"] in worker_cls_ids]

    # Scene condition: auto path already resolved condition via classify_scene_fast
    # + hysteresis above. For manual paths, condition is the operator-selected
    # string; echo it back as scene_condition. scene_reason reflects worker count.
    scene_condition = condition
    scene_reason = f"{valid_worker_count} valid · {suppressed_count} suppressed"

    return {
        "detections": final_detections,
        "valid_workers": valid_workers,
        "valid_worker_count": valid_worker_count,
        "suppressed_count": suppressed_count,
        "raw_worker_count": len(worker_detections),
        "scene_condition": scene_condition,
        "scene_reason": scene_reason,
        "condition": condition,
        "helmets": helmet_detections,
        "vests": vest_detections,
        "used_ensemble": used_ensemble,
        "audit_meta": audit_meta,
    }


# -- Worker temporal confirmation (video export only) ──────────────────────────
class WorkerMemory:
    """
    Require a worker box to appear in ≥ CONFIRM_FRAMES consecutive inference
    cycles before being emitted. After confirmation it is retained until it
    misses ≥ DROP_FRAMES cycles.

    Rationale: cement bags and static construction materials generate a stable
    detection at the same pixel location across ALL frames because they never
    move. However they often generate inconsistent boxes (slightly different
    sizes each inference due to YOLO stochasticity). A real worker that is
    genuinely static (e.g. standing still) will produce highly consistent boxes
    AND will typically have an associated helmet or vest detection nearby.

    Implementation:
      • We track candidate boxes with IOU matching (threshold 0.35).
      • Each candidate accumulates a "seen" counter.
      • Confirmed workers (seen ≥ CONFIRM_FRAMES) are emitted and tracked
        with a "missed" counter.
      • If a confirmed worker disappears for DROP_FRAMES it is removed.
      • PPE presence (has_helmet or has_vest) on any frame immediately
        confirms the detection — real workers have PPE, cement bags do not.
    """
    CONFIRM_FRAMES = 3   # raised 2→3: cement bags pass 2 frames trivially; real
                         # workers with PPE bypass this via ppe_present fast-path
    DROP_FRAMES    = 3   # missed inferences before dropping a confirmed worker
    IOU_THRESH     = 0.30

    def __init__(self):
        self._candidates: list[dict] = []   # unconfirmed
        self._confirmed:  list[dict] = []   # emitted to video

    @staticmethod
    def _best_match(box: list, pool: list):
        """Return (index, iou) of best overlapping item in pool, or (-1, 0)."""
        best_i, best_iou = -1, 0.0
        for i, item in enumerate(pool):
            v = iou_box(
                [c / max(box[2], 1) for c in box[:2]] + [1.0, 1.0],  # simplified
                [c / max(item["box"][2], 1) for c in item["box"][:2]] + [1.0, 1.0],
            )
            # Use absolute-pixel IoU for accuracy
            v = iou_box(box, item["box"])
            if v > best_iou:
                best_iou, best_i = v, i
        return best_i, best_iou

    def update(self, dets: list, worker_cls_ids: set) -> list:
        """
        Feed raw detections for one inference cycle.
        Returns the current set of confirmed worker detections.
        """
        workers = [d for d in dets if d["cls"] in worker_cls_ids]
        others  = [d for d in dets if d["cls"] not in worker_cls_ids]

        matched_cand: set[int] = set()
        matched_conf: set[int] = set()

        # ── Try to match each detected worker to confirmed tracks first ───
        for det in workers:
            ci, ciou = self._best_match(det["box"], self._confirmed)
            if ci >= 0 and ciou >= self.IOU_THRESH and ci not in matched_conf:
                self._confirmed[ci]["box"]    = det["box"]
                self._confirmed[ci]["score"]  = det["score"]
                self._confirmed[ci]["missed"] = 0
                self._confirmed[ci]["payload"] = det
                matched_conf.add(ci)
                continue
            # ── Try to match to candidates ─────────────────────────────────
            ki, kiou = self._best_match(det["box"], self._candidates)
            if ki >= 0 and kiou >= self.IOU_THRESH and ki not in matched_cand:
                self._candidates[ki]["seen"] += 1
                self._candidates[ki]["box"]   = det["box"]
                self._candidates[ki]["payload"] = det
                # PPE presence = immediate confirmation (real worker signal)
                ppe_present = det.get("has_helmet") or det.get("has_vest")
                if self._candidates[ki]["seen"] >= self.CONFIRM_FRAMES or ppe_present:
                    self._confirmed.append({
                        "box":    det["box"],
                        "score":  det["score"],
                        "missed": 0,
                        "payload": det,
                    })
                    self._candidates.pop(ki)
                matched_cand.add(ki)
            else:
                # New candidate
                ppe_present = det.get("has_helmet") or det.get("has_vest")
                if ppe_present:
                    # Immediate confirmation — PPE proves it's a person
                    self._confirmed.append({
                        "box": det["box"], "score": det["score"],
                        "missed": 0, "payload": det,
                    })
                else:
                    self._candidates.append({
                        "box": det["box"], "score": det["score"],
                        "seen": 1, "payload": det,
                    })

        # Age out unmatched confirmed tracks
        for i in range(len(self._confirmed)):
            if i not in matched_conf:
                self._confirmed[i]["missed"] += 1
        self._confirmed = [c for c in self._confirmed if c["missed"] <= self.DROP_FRAMES]

        # Age out stale candidates (missed one cycle → reset or drop)
        new_cands = []
        for i, c in enumerate(self._candidates):
            if i not in matched_cand:
                c["seen"] = max(0, c["seen"] - 1)
                if c["seen"] > 0:
                    new_cands.append(c)
            else:
                new_cands.append(c)
        self._candidates = new_cands

        # Emit confirmed workers + all non-worker detections unchanged
        return [c["payload"] for c in self._confirmed] + others


# ── Server-side IoU tracker (used by /detect/video export) ───────────────────
class ServerSideTracker:
    """
    Lightweight IoU-based tracker for the video export pipeline.

    Purpose: eliminate the "detection flicker" where a box appears/disappears
    between sampled frames. Instead of copying the last raw detection verbatim
    to non-sampled frames, we apply exponential moving-average (EMA) smoothing
    so box positions drift gracefully between inference updates.

    Benefits over raw last-detection reuse:
      • Boxes don't jump when a new inference updates the position.
      • Workers that momentarily drop below conf threshold are carried for up
        to MAX_MISSED inference cycles (faded label) before being dropped.
      • PPE flags (has_helmet, has_vest) are preserved across missing frames.
    """
    MAX_MISSED = 4       # carry-forward cycles before dropping a track
    IOU_THRESH = 0.26    # minimum IoU to match a new det to an existing track
    ALPHA      = 0.45    # EMA weight for new position (higher = snappier)

    def __init__(self):
        self.tracks: list[dict] = []

    def update(self, dets: list) -> list:
        matched: set[int] = set()

        for tr in self.tracks:
            best_iou, best_i = 0.0, -1
            for i, d in enumerate(dets):
                if i in matched or d["cls"] != tr["cls"]:
                    continue
                iou = iou_box(tr["smooth_box"], d["box"])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_i >= 0 and best_iou >= self.IOU_THRESH:
                d = dets[best_i]
                sb, nb = tr["smooth_box"], d["box"]
                tr["smooth_box"] = [sb[j] + (nb[j] - sb[j]) * self.ALPHA for j in range(4)]
                tr["score"]   = d["score"]
                tr["missed"]  = 0
                tr["payload"] = d   # full dict with PPE flags etc.
                matched.add(best_i)
            else:
                tr["missed"] += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.MAX_MISSED]

        for i, d in enumerate(dets):
            if i not in matched:
                self.tracks.append({
                    "cls":        d["cls"],
                    "score":      d["score"],
                    "smooth_box": list(d["box"]),
                    "missed":     0,
                    "payload":    d,
                })

        out = []
        for tr in self.tracks:
            item = dict(tr["payload"])
            item["box"]   = [round(v, 1) for v in tr["smooth_box"]]
            item["score"] = tr["score"]
            out.append(item)
        return out


# ── draw annotations (used only by /detect/video export) ─────────────────────
def draw_detections(img_bgr, detections, mode, condition):
    img = img_bgr.copy()
    h, w = img.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls_name = cls_map.get(det["cls"], "unknown")
        color    = COLORS.get(cls_name, (200, 200, 200))

        label = f"{cls_name} {det['score']:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Banner
    gpu_tag = f"GPU:{DEVICE}" if DEVICE.startswith("cuda") else "CPU"
    banner = f"[BUILDSIGHT | {mode.upper()} | {condition} | {gpu_tag}] {len(detections)} det"
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(img, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

    # Legend
    for i, (cls_id, cls_name) in enumerate(cls_map.items()):
        color = COLORS.get(cls_name, (200, 200, 200))
        lx = 8 + i * 170
        ly = h - 8
        cv2.rectangle(img, (lx, ly - 13), (lx + 13, ly), color, -1)
        cv2.putText(img, cls_name, (lx + 17, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1)

    return img


def img_to_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")


def _build_det_list(detections: list) -> list:
    """Convert raw detection dicts → JSON-serialisable list with class names."""
    out = []
    for d in detections:
        item = {
            "class":      cls_map.get(d["cls"], "unknown"),
            "confidence": round(d["score"], 3),
            "box":        [round(v, 1) for v in d["box"]],
        }
        # Forward PPE association flags (workers only)
        if "has_helmet" in d:
            item["has_helmet"] = d["has_helmet"]
        if "has_vest" in d:
            item["has_vest"] = d["has_vest"]
        out.append(item)
    return out


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status":   "ok",
        "mode":     mode_name,
        "device":   DEVICE,
        "fp16":     USE_HALF,
        "runtime_dir": str(RUNTIME_DIR),
        "model_dir": str(MODEL_DIR),
        "classes":  list(cls_map.values()),
        "ensemble": model_v26 is not None,
        "turner_ai_enabled": mistral_enabled or (ai_model is not None),
        "mistral_enabled": mistral_enabled,
        "mistral_model": MISTRAL_MODEL if mistral_enabled else None,
        "google_api_key_present": bool(AI_API_KEY),
        "gemini_auditor_enabled": gemini_auditor_enabled,
        "gemini_auditor_model": gemini_auditor_model_name,
    }


@app.post("/api/detect/image")
async def detect_image(
    file:       UploadFile = File(...),
    condition:  str   = Form(default="S1_normal"),
    confidence: float = Form(default=PRE_CONF),
    gemini_audit: str = Form(default="0"),
):
    t0   = time.perf_counter()
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # run_inference is CPU/GPU-heavy → offload to threadpool
    use_gemini_audit = gemini_audit not in ("0", "false", "False")
    inference = await run_in_threadpool(
        run_inference, img, condition, confidence, None, None, None, True, 2.0, use_gemini_audit
    )

    detections = inference["detections"]
    used_ensemble = inference["used_ensemble"]
    audit_meta = inference["audit_meta"]
    resolved_condition = inference["condition"]
    det_list     = _build_det_list(detections)
    image_b64    = img_to_b64(img)   # raw image; frontend draws overlays
    elapsed_ms   = round((time.perf_counter() - t0) * 1000)
    class_counts = {}
    for d in det_list:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    # Log metrics to DB
    database.log_metrics(
        worker_count=class_counts.get("worker", 0),
        helmet_count=class_counts.get("helmet", 0),
        vest_count=class_counts.get("safety_vest", 0) or class_counts.get("safety-vest", 0),
        compliance_score=round(sum(1 for d in detections if d.get("has_helmet") and d.get("has_vest")) / max(class_counts.get("worker", 1), 1) * 100, 1),
        condition=resolved_condition
    )

    return {
        "detections":   det_list,
        "class_counts": class_counts,
        "total":        len(det_list),
        "image_b64":    image_b64,
        "mode":         "ensemble-wbf" if used_ensemble else mode_name,
        "condition":    resolved_condition,
        "scene_condition": inference["scene_condition"],
        "scene_reason": inference["scene_reason"],
        "valid_worker_count": inference["valid_worker_count"],
        "raw_worker_count": inference["raw_worker_count"],
        "suppressed_count": inference["suppressed_count"],
        "elapsed_ms":   elapsed_ms,
        "gemini_audit": audit_meta,
    }


@app.post("/api/detect/frame")
async def detect_frame(
    image_b64:     str   = Form(...),
    condition:     str   = Form(default="auto"),   # "auto" | "S1_normal" | S2/S3/S4
    # Per-class thresholds sent as JSON strings from the frontend settings panel.
    # Example: class_conf='{"worker":0.20,"helmet":0.30,"vest":0.18}'
    class_conf:    str   = Form(default="{}"),
    nms_iou:       str   = Form(default="{}"),
    wbf_iou:       str   = Form(default="{}"),
    clahe:         str   = Form(default="1"),
    clahe_clip:    float = Form(default=2.0),
    gemini_audit:  str   = Form(default="0"),
    reset_tracker: str   = Form(default="0"),  # "1" when a new video is loaded
):
    """Video / webcam live mode — base64 frame in, JSON detections out.

    Pass condition="auto" (the new default) to let the backend classify the
    scene automatically.  The resolved condition is echoed in the response as
    "condition" so the frontend can display the live auto-mode indicator.

    Pass reset_tracker="1" when a new video file is loaded to clear the
    hysteresis state so stale S4 readings don't bleed into the new video.
    """
    t0 = time.perf_counter()
    try:
        data = base64.b64decode(image_b64.split(",")[-1])
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return JSONResponse({"error": "Invalid base64 image"}, status_code=400)
    if img is None:
        return JSONResponse({"error": "Could not decode frame"}, status_code=400)

    if reset_tracker not in ("0", "false", "False"):
        _scene_tracker.reset()
        worker_track_state.reset()

    cc  = json.loads(class_conf) if class_conf else {}
    ni  = json.loads(nms_iou)    if nms_iou    else {}
    wi  = json.loads(wbf_iou)    if wbf_iou    else {}
    use_clahe        = clahe not in ("0", "false", "False")
    use_gemini_audit = gemini_audit not in ("0", "false", "False")

    inference = await run_in_threadpool(
        run_inference, img, condition, PRE_CONF,
        cc, ni, wi, use_clahe, clahe_clip, use_gemini_audit,
    )

    detections = inference["detections"]
    used_ensemble = inference["used_ensemble"]
    audit_meta = inference["audit_meta"]
    resolved_condition = inference["condition"]
    det_list   = _build_det_list(detections)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    # Log metrics for analytics
    class_counts = {}
    for d in det_list:
        class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

    database.log_metrics(
        worker_count=class_counts.get("worker", 0),
        helmet_count=class_counts.get("helmet", 0),
        vest_count=class_counts.get("safety_vest", 0) or class_counts.get("safety-vest", 0),
        compliance_score=round(sum(1 for d in detections if d.get("has_helmet") and d.get("has_vest")) / max(class_counts.get("worker", 1), 1) * 100, 1),
        condition=resolved_condition
    )

    return {
        "detections":  det_list,
        "total":       len(det_list),
        "elapsed_ms":  elapsed_ms,
        "mode":        "ensemble-wbf" if used_ensemble else mode_name,
        "condition":   resolved_condition,   # echoed so frontend shows live mode
        "scene_condition": inference["scene_condition"],
        "scene_reason": inference["scene_reason"],
        "valid_worker_count": inference["valid_worker_count"],
        "raw_worker_count": inference["raw_worker_count"],
        "suppressed_count": inference["suppressed_count"],
        "gemini_audit": audit_meta,
    }


@app.post("/api/detect/video")
async def detect_video(
    file:         UploadFile = File(...),
    condition:    str = Form(default="S1_normal"),
    confidence:   float = Form(default=PRE_CONF),
    sample_every: int = Form(default=3),
    gemini_audit: str = Form(default="0"),
):
    """Server-side batch processing → returns annotated MP4 download."""
    data = await file.read()

    use_gemini_audit = gemini_audit not in ("0", "false", "False")

    def _process() -> tuple[str, int, float]:
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_DIR) as tmp_in:
            tmp_in.write(data)
            in_path = tmp_in.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            Path(in_path).unlink(missing_ok=True)
            raise ValueError("Cannot open video file")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=VIDEO_OUTPUT_DIR) as tmp_out:
            out_path = tmp_out.name

        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        t0 = time.perf_counter()
        frame_idx       = 0
        last_detections: list = []
        last_mode       = mode_name
        tracker         = ServerSideTracker()
        wmem            = WorkerMemory()
        worker_cls_ids  = {k for k, v in cls_map.items() if v in ("worker", "person")}

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                inference = run_inference(frame, condition, confidence, gemini_audit=use_gemini_audit)
                raw = inference["detections"]
                used = inference["used_ensemble"]
                _cond = inference["condition"]
                # WorkerMemory gates worker boxes: require ≥2 consecutive
                # inference hits (or PPE presence) before accepting as a
                # real person — eliminates one-shot cement-bag false positives
                confirmed = wmem.update(raw, worker_cls_ids)
                # ServerSideTracker EMA-smooths positions across frames
                last_detections = tracker.update(confirmed)
                last_mode = "ensemble-wbf" if used else mode_name
            annotated = draw_detections(frame, last_detections, last_mode, condition)
            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()
        Path(in_path).unlink(missing_ok=True)
        return out_path, frame_idx, round(time.perf_counter() - t0)

    try:
        out_path, frame_idx, elapsed = await run_in_threadpool(_process)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    stem    = Path(file.filename or "video").stem
    dl_name = f"{stem}_buildsight_detected.mp4"

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=dl_name,
        headers={
            "X-Frames-Processed": str(frame_idx),
            "X-Elapsed-Seconds":  str(elapsed),
            "X-Device":           DEVICE,
            "X-Condition":        condition,
        },
        background=BackgroundTask(lambda p=out_path: Path(p).unlink(missing_ok=True)),
    )


# ── Turner AI Assistant Route ─────────────────────────────────────────────────
# Primary:  Mistral AI  (MISTRAL_API_KEY)
# Fallback: Google Gemini (GOOGLE_API_KEY)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ai/chat")
async def ai_chat(req: ChatRequest):
    request_id = uuid.uuid4().hex[:10]

    if not mistral_enabled and not ai_model:
        log_turner_event(
            "turner_request_rejected",
            request_id=request_id,
            reason="MISSING_API_KEY",
        )
        return JSONResponse({
            "response": (
                "Turner AI is currently offline. "
                "Set MISTRAL_API_KEY (or GOOGLE_API_KEY) in the backend environment."
            ),
            "error": "MISSING_API_KEY",
            "request_id": request_id,
        }, status_code=503)

    full_prompt = _build_site_prompt(req)
    ctx = req.context
    log_turner_event(
        "turner_request_received",
        request_id=request_id,
        provider="mistral" if mistral_enabled else "gemini",
        message=req.message,
        history_length=len(req.history),
        context={
            "workers": ctx.get("workers", 0),
            "helmets": ctx.get("helmets", 0),
            "vests":   ctx.get("vests", 0),
            "condition": ctx.get("condition", "Unknown"),
            "alerts": ctx.get("alerts", []),
        },
    )

    # ── Mistral path ─────────────────────────────────────────────────────────
    if mistral_enabled:
        try:
            messages = _build_mistral_messages(req, full_prompt)
            response_text = await run_in_threadpool(_call_mistral_sync, messages)
            log_turner_event("turner_response_success", request_id=request_id,
                             provider="mistral", response=response_text)
            return {"response": response_text, "status": "success",
                    "provider": "mistral", "request_id": request_id}
        except Exception as e:
            logger.warning("Mistral call failed (%s): %s — trying Gemini fallback.", type(e).__name__, e)
            if not ai_model:
                log_turner_event("turner_response_error", request_id=request_id,
                                 error=str(e), error_type=type(e).__name__)
                return JSONResponse({
                    "response": "Turner encountered a Mistral error and no Gemini fallback is configured. Please retry.",
                    "error": type(e).__name__,
                    "request_id": request_id,
                }, status_code=500)

    # ── Gemini fallback path ─────────────────────────────────────────────────
    try:
        gemini_history = [
            {"role": "user" if m.role == "user" else "model", "parts": [m.content]}
            for m in req.history
        ]
        chat = ai_model.start_chat(history=gemini_history)
        response = await run_in_threadpool(chat.send_message, full_prompt)
        response_text = _extract_response_text(response)
        block_metadata = _response_block_metadata(response)

        if not response_text:
            blocked = bool(block_metadata.get("prompt_block_reason")) or any(
                reason and "SAFETY" in reason.upper()
                for reason in block_metadata.get("candidate_finish_reasons", [])
            )
            if blocked:
                return JSONResponse({
                    "response": _turner_blocked_message(block_metadata),
                    "status": "blocked", "error": "SAFETY_BLOCKED",
                    "request_id": request_id, "details": block_metadata,
                }, status_code=200)
            return JSONResponse({
                "response": "I did not receive a usable model response. Please retry.",
                "status": "empty", "error": "EMPTY_RESPONSE",
                "request_id": request_id,
            }, status_code=502)

        log_turner_event("turner_response_success", request_id=request_id,
                         provider="gemini", response=response_text)
        return {"response": response_text, "status": "success",
                "provider": "gemini", "request_id": request_id}

    except Exception as e:
        log_turner_event("turner_response_error", request_id=request_id,
                         error=str(e), error_type=type(e).__name__)
        return JSONResponse({
            "response": "I encountered a synchronization error. Please try again.",
            "error": type(e).__name__,
            "request_id": request_id,
        }, status_code=500)


# ── Turner AI Streaming Route ─────────────────────────────────────────────────
# Uses Mistral's native SSE streaming so tokens appear incrementally in the UI.
# Requires: MISTRAL_API_KEY  (Gemini does not support the same SSE protocol)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/ai/chat/stream")
async def ai_chat_stream(req: ChatRequest):
    request_id = uuid.uuid4().hex[:10]

    if not mistral_enabled:
        # Graceful degradation: fall back to the non-streaming route
        return JSONResponse({
            "response": (
                "Streaming requires MISTRAL_API_KEY. "
                "Use /api/ai/chat for non-streaming responses."
            ),
            "error": "STREAMING_UNAVAILABLE",
            "request_id": request_id,
        }, status_code=503)

    full_prompt = _build_site_prompt(req)
    messages    = _build_mistral_messages(req, full_prompt)

    async def _event_stream():
        """Yield SSE-formatted tokens from Mistral's streaming API."""
        try:
            import httpx  # preferred async client; falls back to sync generator below
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST",
                    MISTRAL_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream",
                    },
                    json={"model": MISTRAL_MODEL, "messages": messages, "stream": True},
                ) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            yield "data: [DONE]\n\n"
                            return
                        try:
                            obj   = json.loads(payload)
                            token = obj["choices"][0]["delta"].get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token, 'request_id': request_id})}\n\n"
                        except (KeyError, json.JSONDecodeError):
                            pass
        except ImportError:
            # httpx not installed — call sync Mistral and emit full response as
            # a single SSE event so the frontend still receives a valid stream.
            try:
                full_text = await run_in_threadpool(_call_mistral_sync, messages)
                yield f"data: {json.dumps({'token': full_text, 'request_id': request_id})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e), 'request_id': request_id})}\n\n"
        except Exception as e:
            logger.error("Streaming error (%s): %s", type(e).__name__, e)
            yield f"data: {json.dumps({'error': str(e), 'request_id': request_id})}\n\n"

    log_turner_event("turner_stream_started", request_id=request_id,
                     message=req.message, history_length=len(req.history))
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )



# ── Configuration & Sync Endpoints ───────────────────────────────────────────

@app.get("/api/settings/threshold")
async def get_threshold():
    return {"threshold": THRESHOLD_STATE["global_floor"]}

@app.post("/api/settings/threshold")
async def update_threshold(req: dict):
    new_val = req.get("threshold")
    if new_val is not None:
        updated, _ = update_thresholds(global_floor=float(new_val))
        _scene_tracker.invalidate_cache()
        return {"status": "success", "threshold": updated["global_floor"]}
    return JSONResponse(status_code=400, content={"error": "Missing threshold value"})


@app.get("/api/thresholds")
async def get_thresholds():
    return THRESHOLD_STATE


@app.post("/api/thresholds")
async def set_thresholds(req: dict):
    updated, warnings = update_thresholds(
        global_floor=req.get("global_floor"),
        worker=req.get("worker"),
        helmet=req.get("helmet"),
        vest=req.get("vest"),
    )
    _scene_tracker.invalidate_cache()
    return {
        "status": "ok",
        "thresholds": updated,
        "warnings": warnings,
        "scene_condition": _scene_tracker.current_condition,
        "scene_reason": _scene_tracker.current_reason,
    }


@app.post("/api/thresholds/reset")
async def reset_thresholds():
    """Restore all confidence thresholds to production-calibrated defaults."""
    restored = reset_thresholds_to_defaults()
    return {
        "status": "reset_to_defaults",
        "thresholds": restored,
        "warnings": [],
        "scene_condition": _scene_tracker.current_condition,
        "scene_reason": _scene_tracker.current_reason,
    }

# ── Analytics Endpoints ──────────────────────────────────────────────────────

@app.get("/api/analytics/summary")
async def get_compliance_summary(days: int = 7):
    data = database.get_analytics_summary(days)
    return {"summary": data}

@app.get("/api/analytics/history")
async def get_detection_history(limit: int = 100):
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM metrics ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, log_level=BACKEND_LOG_LEVEL)
