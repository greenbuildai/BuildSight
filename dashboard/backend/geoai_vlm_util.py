"""
BuildSight GeoAI — VLM Activity Description Utility (Moondream2)
================================================================
Async-safe, lazily-loaded Moondream2 wrapper for site activity narration.

Behaviour:
  - On first call, tries to load `vikhyatk/moondream2` from HuggingFace Hub
    (or a local cache).  Requires `transformers`, `torch`, and `Pillow`.
  - If unavailable, generates rule-based descriptions from detection stats.
  - The last description is cached for VLM_CACHE_TTL_S seconds so the
    model is not called more than once per interval.
  - Exposes both a sync and an async wrapper for use inside FastAPI routes.

Usage:
    from geoai_vlm_util import describe_frame_async, get_cached_entry

    entry = await describe_frame_async(jpeg_bytes=frame_jpeg, fallback_stats=stats)
    # entry = {"description": "...", "timestamp": 1234567890.0, "source": "moondream2"}
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("BuildSight.VLM")

# ── Config ───────────────────────────────────────────────────────────────────
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"
VLM_CACHE_TTL_S = 10.0

DEFAULT_QUESTION = (
    "Describe the construction site activity in this image. "
    "Note the number of workers visible, whether they are wearing helmets and "
    "safety vests, and any obvious safety risks or hazards."
)

# ── Lazy model state ─────────────────────────────────────────────────────────
_VLM_MODEL: Optional[Any] = None
_VLM_TOKENIZER: Optional[Any] = None
_VLM_AVAILABLE: Optional[bool] = None  # None = not yet checked

# ── Cached result ─────────────────────────────────────────────────────────────
_CACHED_ENTRY: Dict = {}
_CACHED_TS: float = 0.0


# ── Model loader ──────────────────────────────────────────────────────────────

def _try_load_moondream() -> bool:
    """Load Moondream2 once. Returns True on success."""
    global _VLM_MODEL, _VLM_TOKENIZER, _VLM_AVAILABLE
    if _VLM_AVAILABLE is not None:
        return _VLM_AVAILABLE

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        import torch

        log.info("VLM: Loading Moondream2 (%s r=%s) — first load may be slow…",
                 VLM_MODEL_ID, VLM_REVISION)

        tok = AutoTokenizer.from_pretrained(VLM_MODEL_ID, revision=VLM_REVISION)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_ID,
            trust_remote_code=True,
            revision=VLM_REVISION,
            torch_dtype=dtype,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        _VLM_MODEL = model
        _VLM_TOKENIZER = tok
        _VLM_AVAILABLE = True
        log.info("VLM: Moondream2 ready on %s", device)
        return True

    except ImportError as exc:
        log.info(
            "VLM: Required package not found (%s). "
            "Run `pip install transformers torch Pillow` to enable Moondream2. "
            "Rule-based fallback active.",
            exc,
        )
        _VLM_AVAILABLE = False
        return False
    except Exception as exc:
        log.warning("VLM: Moondream2 load failed (%s) — rule-based fallback active.", exc)
        _VLM_AVAILABLE = False
        return False


# ── Inference ─────────────────────────────────────────────────────────────────

def describe_frame_sync(
    frame_bgr: Optional[np.ndarray] = None,
    jpeg_bytes: Optional[bytes] = None,
    question: str = DEFAULT_QUESTION,
    point_prompt: Optional[Tuple[int, int]] = None,
    fallback_stats: Optional[Dict] = None,
    force_refresh: bool = False,
) -> Dict:
    """
    Describe a frame using Moondream2, or fall back to rule-based text.
    If point_prompt is provided, the VLM focuses on that specific spatial area.
    """
    global _CACHED_ENTRY, _CACHED_TS
    now = time.time()

    # We only cache general descriptions. Point-prompts are always unique.
    if not point_prompt and not force_refresh and _CACHED_ENTRY and (now - _CACHED_TS) < VLM_CACHE_TTL_S:
        return _CACHED_ENTRY

    description = ""
    source = "rule_based"

    # ── Moondream2 path ───────────────────────────────────────────────────────
    if _try_load_moondream() and (frame_bgr is not None or jpeg_bytes is not None):
        try:
            from PIL import Image
            import io

            if jpeg_bytes:
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            else:
                import cv2
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

            # If we have a point, we might want to crop or highlight it.
            # V1: Smart cropping to increase VLM performance on small objects
            if point_prompt:
                px, py = point_prompt
                w, h = img.size
                crop_size = 400
                left = max(0, px - crop_size // 2)
                top = max(0, py - crop_size // 2)
                right = min(w, left + crop_size)
                bottom = min(h, top + crop_size)
                
                # Ensure we don't go out of bounds if point is near bottom/right
                if right == w: left = max(0, w - crop_size)
                if bottom == h: top = max(0, h - crop_size)
                
                img = img.crop((left, top, right, bottom))
                if "visible" not in question.lower():
                    question = f"Describe specifically what is at the center of this focused view. {question}"
            
            enc = _VLM_MODEL.encode_image(img)
            answer = _VLM_MODEL.answer_question(enc, question, _VLM_TOKENIZER)
            description = answer.strip()
            source = "moondream2"
            log.debug("VLM: spatial query completed (%d chars)", len(description))

        except Exception as exc:
            log.error("VLM: spatial inference error — %s", exc)

    # ── Rule-based fallback ───────────────────────────────────────────────────
    if not description:
        description = _rule_based_description(fallback_stats or {})
        source = "rule_based"

    entry = {
        "description": description,
        "timestamp": now,
        "source": source,
        "question": question,
        "point": point_prompt
    }
    
    if not point_prompt:
        _CACHED_ENTRY = entry
        _CACHED_TS = now
        
    return entry


def _rule_based_description(stats: Dict) -> str:
    """Generate plausible text from detection counts when the VLM is not available."""
    workers = stats.get("total_workers", 0) or stats.get("active_workers", 0)
    helmets = stats.get("helmets_detected", 0)
    vests = stats.get("vests_detected", 0)
    violations = stats.get("proximity_violations", 0)
    scene = stats.get("scene", "S1_normal")
    avg_risk = stats.get("avg_site_risk", 0.0)

    if workers == 0:
        return "No workers currently detected on site. Site appears clear."

    helmet_pct = int(helmets / max(workers, 1) * 100)
    vest_pct = int(vests / max(workers, 1) * 100)
    parts = [f"{workers} worker{'s' if workers != 1 else ''} detected on site."]

    if helmet_pct == 100:
        parts.append("All workers wearing helmets.")
    elif helmet_pct >= 80:
        parts.append(f"Helmet compliance: {helmet_pct}%.")
    else:
        parts.append(
            f"Low helmet compliance — only {helmet_pct}% of workers wearing helmets."
        )

    if vest_pct < 60:
        parts.append(f"Safety vests missing on {100 - vest_pct}% of workers.")
    elif vest_pct < 100:
        parts.append(f"Vest compliance: {vest_pct}%.")

    if violations > 0:
        parts.append(
            f"{violations} proximity violation{'s' if violations != 1 else ''} detected."
        )

    if "S4" in scene:
        parts.append("Crowded site — elevated density monitoring active.")
    elif "S3" in scene:
        parts.append("Low-light conditions — visibility may be impaired.")
    elif "S2" in scene:
        parts.append("Dusty conditions — PPE verification confidence reduced.")

    if avg_risk > 0.7:
        parts.append("Overall site risk is HIGH.")
    elif avg_risk > 0.4:
        parts.append("Moderate site risk level.")

    return " ".join(parts)


def get_cached_entry() -> Tuple[Dict, bool]:
    """Return (entry_dict, is_stale)."""
    stale = not _CACHED_ENTRY or (time.time() - _CACHED_TS) > VLM_CACHE_TTL_S
    return _CACHED_ENTRY, stale


def is_available() -> bool:
    """Return True if Moondream2 is loaded and ready."""
    return bool(_VLM_AVAILABLE)


# ── Async wrapper ─────────────────────────────────────────────────────────────

async def describe_frame_async(
    frame_bgr: Optional[np.ndarray] = None,
    jpeg_bytes: Optional[bytes] = None,
    question: str = DEFAULT_QUESTION,
    point_prompt: Optional[Tuple[int, int]] = None,
    fallback_stats: Optional[Dict] = None,
    force_refresh: bool = False,
) -> Dict:
    """Async wrapper — offloads blocking VLM inference to the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: describe_frame_sync(
            frame_bgr, jpeg_bytes, question, point_prompt, fallback_stats, force_refresh
        ),
    )
