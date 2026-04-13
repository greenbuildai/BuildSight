"""
BuildSight GeoAI — SAM Segmentation Utility
============================================
Async-safe, lazily-loaded SAM wrapper for automatic zone delineation.

Behaviour:
  - If `segment_anything` is available and a local checkpoint exists,
    loads SAM-ViT-B once and caches it.
  - If unavailable (no package, no checkpoint), returns [] silently.
  - Accepts a BGR numpy array and optional pixel-space point prompts
    (worker centroids from the detection pipeline).
  - Returns a list of GeoJSON-compatible Feature dicts with normalised
    [0,1] polygon coordinates suitable for the frontend overlay.
  - Results are frame-hash-cached for CACHE_TTL_S seconds to avoid
    re-segmenting identical frames.

Usage:
    from geoai_sam_util import segment_frame_async, segment_frame_sync

    zones = await segment_frame_async(frame_bgr, point_prompts=worker_centroids)
    # or synchronously:
    zones = segment_frame_sync(frame_bgr)
"""

import asyncio
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("BuildSight.SAM")

# ── Lazy model state ────────────────────────────────────────────────────────
_SAM_MODEL: Optional[Any] = None
_SAM_PREDICTOR: Optional[Any] = None
_SAM_AVAILABLE: Optional[bool] = None  # None = not yet checked

# ── Result cache ────────────────────────────────────────────────────────────
_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}
_CACHE_TTL_S = 30.0

# ── Local checkpoint search paths ───────────────────────────────────────────
_CKPT_CANDIDATES = [
    "weights/sam_vit_b.pth",
    "../weights/sam_vit_b.pth",
    os.path.join(os.path.expanduser("~"), ".cache", "sam", "sam_vit_b.pth"),
    os.path.join(os.path.expanduser("~"), ".cache", "buildsight", "sam_vit_b.pth"),
]


# ── Model loader ────────────────────────────────────────────────────────────

def _try_load_sam() -> bool:
    """Attempt to load SAM once. Returns True on success, False permanently on failure."""
    global _SAM_MODEL, _SAM_PREDICTOR, _SAM_AVAILABLE
    if _SAM_AVAILABLE is not None:
        return _SAM_AVAILABLE

    try:
        from segment_anything import sam_model_registry, SamPredictor  # type: ignore
        import torch

        ckpt = next((p for p in _CKPT_CANDIDATES if os.path.exists(p)), None)
        if ckpt is None:
            log.info(
                "SAM: No local checkpoint found at any of %s — "
                "place sam_vit_b.pth in dashboard/backend/weights/ to enable. "
                "SAM features disabled.",
                _CKPT_CANDIDATES,
            )
            _SAM_AVAILABLE = False
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("SAM: Loading ViT-B from %s on %s …", ckpt, device)
        sam = sam_model_registry["vit_b"](checkpoint=ckpt)
        sam.to(device=device)
        sam.eval()

        _SAM_MODEL = sam
        _SAM_PREDICTOR = SamPredictor(sam)
        _SAM_AVAILABLE = True
        log.info("SAM: Ready on %s", device)
        return True

    except ImportError:
        log.info(
            "SAM: `segment_anything` package not installed — "
            "run `pip install segment-anything` to enable. SAM features disabled."
        )
        _SAM_AVAILABLE = False
        return False
    except Exception as exc:
        log.warning("SAM: Load failed (%s) — SAM features disabled.", exc)
        _SAM_AVAILABLE = False
        return False


# ── Mask → polygon ──────────────────────────────────────────────────────────

def _mask_to_normalised_polygon(
    mask: np.ndarray, img_w: int, img_h: int
) -> Optional[List[List[float]]]:
    """Convert a binary mask to a simplified, closed, normalised polygon."""
    try:
        import cv2  # type: ignore

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 400:  # skip tiny masks
            return None
        epsilon = 0.015 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            return None
        coords = [[round(pt[0] / img_w, 5), round(pt[1] / img_h, 5)] for pt in approx]
        coords.append(coords[0])  # close ring
        return coords
    except Exception:
        return None


# ── Frame hash ───────────────────────────────────────────────────────────────

def _frame_hash(frame: np.ndarray) -> str:
    sub = frame[::16, ::16]
    return hashlib.md5(sub.tobytes()).hexdigest()[:16]


# ── Core sync function ───────────────────────────────────────────────────────

def segment_frame_sync(
    frame_bgr: np.ndarray,
    point_prompts: Optional[List[Tuple[int, int]]] = None,
    min_score: float = 0.70,
) -> List[Dict]:
    """
    Segment a BGR frame with SAM and return GeoJSON feature dicts.

    Args:
        frame_bgr:      OpenCV BGR numpy array.
        point_prompts:  Optional list of (x, y) pixel coords as foreground prompts.
                        If None, a 3×3 grid of auto-prompts is used.
        min_score:      Minimum SAM mask quality score to include.

    Returns:
        List of GeoJSON Feature dicts.  Empty list if SAM not available.
    """
    if not _try_load_sam():
        return []

    fhash = _frame_hash(frame_bgr)
    now = time.time()

    if fhash in _CACHE:
        ts, result = _CACHE[fhash]
        if now - ts < _CACHE_TTL_S:
            return result

    try:
        rgb = frame_bgr[:, :, ::-1].copy()
        h, w = rgb.shape[:2]

        _SAM_PREDICTOR.set_image(rgb)

        if point_prompts:
            pts = np.array(point_prompts, dtype=np.float32)
        else:
            pts = np.array(
                [[w * x / 4, h * y / 4] for x in range(1, 4) for y in range(1, 4)],
                dtype=np.float32,
            )
        labels = np.ones(len(pts), dtype=np.int32)

        masks, scores, _ = _SAM_PREDICTOR.predict(
            point_coords=pts,
            point_labels=labels,
            multimask_output=True,
        )

        features: List[Dict] = []
        for mask, score in zip(masks, scores):
            if float(score) < min_score:
                continue
            coords = _mask_to_normalised_polygon(mask, w, h)
            if coords is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "source": "SAM",
                        "confidence": round(float(score), 4),
                        "zone": "sam_segment",
                        "risk": "none",
                        "auto": True,
                    },
                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                }
            )

        _CACHE[fhash] = (now, features)
        log.info("SAM: %d zone(s) extracted (score≥%.2f)", len(features), min_score)
        return features

    except Exception as exc:
        log.error("SAM segmentation error: %s", exc)
        return []


def is_available() -> bool:
    """Return True if SAM is loaded and ready."""
    return bool(_SAM_AVAILABLE)


# ── Async wrapper ────────────────────────────────────────────────────────────

async def segment_frame_async(
    frame_bgr: np.ndarray,
    point_prompts: Optional[List[Tuple[int, int]]] = None,
    min_score: float = 0.70,
) -> List[Dict]:
    """Async wrapper — offloads blocking SAM inference to the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: segment_frame_sync(frame_bgr, point_prompts, min_score),
    )
