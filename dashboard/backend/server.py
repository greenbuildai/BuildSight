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

# ── Condition-specific post-WBF confidence gates ───────────────────────────────
# In low-light and dusty scenes every class scores lower, so we relax the gates
# rather than miss real detections. Crowded scenes get a slightly lower worker
# gate so distant/small workers still pass.
# Keys: SASTRA class IDs  0=helmet, 1=safety_vest, 2=worker
POST_WBF_BY_CONDITION: dict[str, dict[int, float]] = {
    "S1_normal":    {0: 0.30, 1: 0.14, 2: 0.18},
    "S2_dusty":     {0: 0.25, 1: 0.10, 2: 0.15},  # vest hard to see in dust
    "S3_low_light": {0: 0.22, 1: 0.10, 2: 0.14},  # everything scores weaker
    "S4_crowded":   {0: 0.28, 1: 0.12, 2: 0.15},  # small/distant workers
}

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
AI_API_KEY = os.environ.get("GOOGLE_API_KEY")
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
MIN_HELMET_CONF = 0.35

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
    min_h_frac = 0.06 if condition == "S4_crowded" else 0.08
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
    size_frac = 0.010 if condition in ("S4_crowded", "S3_low_light") else 0.012
    min_px    = min(img_w, img_h) * size_frac
    if bw < min_px or bh < min_px:
        return False

    # ── Signal 3: single-model agreement gate ─────────────────────────────
    if model_count == 1:
        if condition == "S3_low_light":
            single_thresh = 0.32
        elif condition == "S2_dusty":
            single_thresh = 0.35
        elif condition == "S4_crowded":
            single_thresh = 0.40
        else:
            single_thresh = 0.50   # S1_normal: cement bags rarely score > 0.50 solo
        if score < single_thresh:
            return False

    # ── Signal 5: portrait check (applies regardless of model_count) ───────
    # Calibrated per-condition — catches boxes that fool BOTH models:
    #   S1_normal:   aspect > 0.88 and score < 0.70 → reject (bags on planks)
    #   S2_dusty:    aspect > 0.95 and score < 0.65 → reject (haze inflates boxes)
    #   S3_low_light: aspect > 1.00 and score < 0.62 → reject (dark scene noise)
    #   S4_crowded:  skip — overlapping workers produce legitimately wide boxes
    if condition == "S1_normal":
        if aspect > 0.88 and score < 0.70:
            return False
    elif condition == "S2_dusty":
        if aspect > 0.95 and score < 0.65:
            return False
    elif condition == "S3_low_light":
        if aspect > 1.00 and score < 0.62:
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
            if hd.get("score", 1.0) < MIN_HELMET_CONF:
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

    return result


# ── Class-ID helpers (module-level, computed once after load_models) ──────────
def _worker_ids() -> set:
    return {k for k, v in cls_map.items() if v in ("worker", "person")}

def _helmet_ids() -> set:
    return {k for k, v in cls_map.items() if v == "helmet"}

def _vest_ids() -> set:
    return {k for k, v in cls_map.items() if v in ("safety_vest", "safety-vest")}


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


# ── inference core (runs in threadpool — never blocks the event loop) ─────────
def run_inference(
    img_bgr:     np.ndarray,
    condition:   str   = "S1_normal",
    conf_thresh: float = PRE_CONF,
    # Per-class conf gates applied POST-WBF (frontend sends these as JSON dicts)
    class_conf:  dict | None = None,   # {worker, helmet, vest} → float
    nms_iou_map: dict | None = None,   # {worker, helmet, vest} → float
    wbf_iou_map: dict | None = None,   # {worker, helmet, vest} → float
    clahe:       bool  = True,
    clahe_clip:  float = 2.0,
):
    # ── Condition-aware preprocessing ────────────────────────────────────────
    # All conditions get CLAHE. S3/S2 get additional filters.
    # S3_low_light: boost CLAHE clip to 3.0 (darker scene = more contrast needed)
    # S2_dusty: use clip 2.5 + bilateral filter (enhance_frame handles both)
    if clahe:
        effective_clip = clahe_clip
        if condition == "S3_low_light" and clahe_clip < 3.0:
            effective_clip = 3.0   # stronger boost for dark footage
        elif condition == "S2_dusty" and clahe_clip < 2.5:
            effective_clip = 2.5   # moderate boost + bilateral (in enhance_frame)
        img_bgr = enhance_frame(img_bgr, condition, effective_clip)

    h, w = img_bgr.shape[:2]

    # ── Condition-aware pre-WBF confidence threshold ──────────────────────────
    # Lower pre_conf for difficult conditions: weak-but-real detections (small
    # workers in crowded frames, low-contrast workers in dark/dusty scenes) must
    # reach WBF so both models can reinforce them. The post-WBF gate filters
    # the false positives that pass through at lower pre_conf.
    if condition == "S4_crowded":
        pre_conf = min(conf_thresh, 0.15)   # catch small/distant workers
    elif condition == "S3_low_light":
        pre_conf = min(conf_thresh, 0.15)   # everything scores weaker in dark
    elif condition == "S2_dusty":
        pre_conf = min(conf_thresh, 0.17)   # dust attenuates scores
    else:
        pre_conf = conf_thresh              # S1_normal: use value as-is

    # ── Per-condition WBF IoU + NMS ───────────────────────────────────────────
    # S4_crowded:
    #   Worker WBF IoU = 0.72 — only near-identical boxes fuse; adjacent workers
    #   (IoU 0.20-0.55) stay separate. NMS IoU = 0.65 so per-model NMS doesn't
    #   suppress nearby workers before WBF sees them.
    # S3_low_light:
    #   Worker WBF IoU lowered (0.60) — noisy boxes shift slightly frame-to-frame
    #   so we allow slightly looser merging. NMS standard.
    # S1_normal / S2_dusty: standard params.
    if condition == "S4_crowded":
        wbf_iou_cond = {0: 0.45, 1: 0.50, 2: 0.72}
        base_nms_iou = 0.65
    elif condition == "S3_low_light":
        wbf_iou_cond = {0: 0.40, 1: 0.45, 2: 0.60}
        base_nms_iou = NMS_IOU
    else:
        wbf_iou_cond = dict(WBF_IOU)
        base_nms_iou = NMS_IOU

    # Condition-specific post-WBF confidence gate (relaxed in dark/dusty scenes)
    cond_conf_gate = POST_WBF_BY_CONDITION.get(condition, POST_WBF_GLOBAL)

    # ── Apply frontend per-class overrides ────────────────────────────────────
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

    # NMS IoU: use the most permissive (highest) value across all classes so
    # no class gets over-suppressed within each individual model's predict().
    if nms_iou_map:
        nms_iou = max(float(v) for v in nms_iou_map.values())
    else:
        nms_iou = base_nms_iou

    # ── YOLOv11 (primary) ────────────────────────────────────────────────────
    r11 = model_v11.predict(
        img_bgr, device=DEVICE, verbose=False,
        conf=pre_conf, iou=nms_iou, half=USE_HALF,
        agnostic_nms=False,   # class-aware NMS — helmet never suppresses worker
    )[0]
    boxes11, scores11, labels11 = [], [], []
    for box in (r11.boxes or []):
        boxes11.append(box.xyxy[0].cpu().numpy().tolist())
        scores11.append(float(box.conf[0]))
        labels11.append(int(box.cls[0]))

    # ── Early-exit check or full WBF ─────────────────────────────────────────
    # Crowded scenes always run full ensemble — more workers means more cases
    # where one model misses, so early-exit is disabled for S4_crowded.
    if model_v26 is not None:
        force_ensemble = (condition == "S4_crowded")
        early_exit = (
            not force_ensemble and
            len(scores11) >= 1 and
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
            raw = wbf_fuse(
                [(boxes11, scores11, labels11), (boxes26, scores26, labels26)],
                w, h, iou_override=wbf_iou_cond, conf_gate=cond_conf_gate,
            )
            used_ensemble = True
    else:
        raw = [{"box": b, "score": s, "cls": l, "n_models": 1}
               for b, s, l in zip(boxes11, scores11, labels11)]
        used_ensemble = False

    # ── Worker false-positive filter ──────────────────────────────────────────
    # IMPORTANT: PPE proximity is checked HERE, before the worker filter, so
    # is_valid_worker can skip geometry checks for boxes that have a confirmed
    # helmet or vest spatially adjacent (PPE = definitive person signal).
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

    # ── Per-class confidence filter (post-WBF, user-tunable) ─────────────────
    if class_conf:
        fallback = cond_conf_gate
        def _passes(d: dict) -> bool:
            cls_name = cls_map.get(d["cls"], "")
            if cls_name in ("worker", "person"):
                return d["score"] >= float(class_conf.get("worker", fallback.get(d["cls"], 0.18)))
            if cls_name == "helmet":
                return d["score"] >= float(class_conf.get("helmet", fallback.get(d["cls"], 0.30)))
            if cls_name in ("safety_vest", "safety-vest"):
                return d["score"] >= float(class_conf.get("vest", fallback.get(d["cls"], 0.14)))
            return True
        raw = [d for d in raw if _passes(d)]

    # ── Worker-to-PPE association (post-WBF, pre-return) ─────────────────────
    raw = associate_ppe_to_workers(raw)

    return raw, used_ensemble


# ── Worker temporal confirmation (video export only) ──────────────────────────
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
    IOU_THRESH = 0.25    # minimum IoU to match a new det to an existing track
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
    }


@app.post("/api/detect/image")
async def detect_image(
    file:       UploadFile = File(...),
    condition:  str   = Form(default="S1_normal"),
    confidence: float = Form(default=PRE_CONF),
):
    t0   = time.perf_counter()
    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image"}, status_code=400)

    # run_inference is CPU/GPU-heavy → offload to threadpool
    detections, used_ensemble = await run_in_threadpool(
        run_inference, img, condition, confidence
    )

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
        condition=condition
    )

    return {
        "detections":   det_list,
        "class_counts": class_counts,
        "total":        len(det_list),
        "image_b64":    image_b64,
        "mode":         "ensemble-wbf" if used_ensemble else mode_name,
        "condition":    condition,
        "elapsed_ms":   elapsed_ms,
    }


@app.post("/api/detect/frame")
async def detect_frame(
    image_b64:  str   = Form(...),
    condition:  str   = Form(default="S1_normal"),
    # Per-class thresholds sent as JSON strings from the frontend settings panel.
    # Example: class_conf='{"worker":0.20,"helmet":0.30,"vest":0.18}'
    class_conf: str   = Form(default="{}"),
    nms_iou:    str   = Form(default="{}"),
    wbf_iou:    str   = Form(default="{}"),
    clahe:      str   = Form(default="1"),
    clahe_clip: float = Form(default=2.0),
):
    """Video / webcam live mode — base64 frame in, JSON detections out.
    The frontend draws overlays; we never re-encode an annotated image here."""
    t0 = time.perf_counter()
    try:
        data = base64.b64decode(image_b64.split(",")[-1])
        arr  = np.frombuffer(data, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return JSONResponse({"error": "Invalid base64 image"}, status_code=400)
    if img is None:
        return JSONResponse({"error": "Could not decode frame"}, status_code=400)

    cc  = json.loads(class_conf) if class_conf else {}
    ni  = json.loads(nms_iou)    if nms_iou    else {}
    wi  = json.loads(wbf_iou)    if wbf_iou    else {}
    use_clahe = clahe not in ("0", "false", "False")

    detections, used_ensemble = await run_in_threadpool(
        run_inference, img, condition, PRE_CONF,
        cc, ni, wi, use_clahe, clahe_clip,
    )

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
        condition=condition
    )

    return {
        "detections": det_list,
        "total":      len(det_list),
        "elapsed_ms": elapsed_ms,
        "mode":       "ensemble-wbf" if used_ensemble else mode_name,
    }


@app.post("/api/detect/video")
async def detect_video(
    file:         UploadFile = File(...),
    condition:    str = Form(default="S1_normal"),
    confidence:   float = Form(default=PRE_CONF),
    sample_every: int = Form(default=3),
):
    """Server-side batch processing → returns annotated MP4 download."""
    data = await file.read()

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
                raw, used = run_inference(frame, condition, confidence)
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
    return {"threshold": PRE_CONF}

@app.post("/api/settings/threshold")
async def update_threshold(req: dict):
    global PRE_CONF
    new_val = req.get("threshold")
    if new_val is not None:
        PRE_CONF = float(new_val)
        logger.info("Updated PRE_CONF threshold to: %s", PRE_CONF)
        return {"status": "success", "threshold": PRE_CONF}
    return JSONResponse(status_code=400, content={"error": "Missing threshold value"})

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
