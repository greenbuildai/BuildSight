"""
BuildSight GeoAI — VLM Activity Description Utility (Florence-2)
================================================================
Async-safe Florence-2 wrapper for site activity narration.

Load order:
  1. florence-community/Florence-2-base-ft  (fine-tuned, preferred)
  2. microsoft/Florence-2-base              (base, already cached by download_models.py)
  Falls back to rule-based descriptions if both fail.

Retry policy: failed loads are retried after VLM_RETRY_INTERVAL_S (300s) so a
transient download failure doesn't permanently disable the VLM.

Startup: call trigger_preload() from the server startup event to warm the model
before the first user request.
"""

# ── CRITICAL: Mock flash_attn BEFORE any transformers imports ─────────────────
# Florence-2 has a soft dependency on flash_attn which doesn't build on Windows.
# Newer transformers also checks PACKAGE_DISTRIBUTION_MAPPING["flash_attn"] at
# import time — patch that dict too so the KeyError doesn't block loading.
import sys
import types
from unittest.mock import MagicMock

if "flash_attn" not in sys.modules:
    _fa = types.ModuleType("flash_attn")
    _fa.__spec__ = MagicMock()
    sys.modules["flash_attn"] = _fa

# Patch transformers' internal package-distribution map so is_flash_attn_*_available()
# doesn't raise a KeyError when flash_attn is absent on Windows.
try:
    import transformers.utils.import_utils as _triu
    if hasattr(_triu, "PACKAGE_DISTRIBUTION_MAPPING"):
        _triu.PACKAGE_DISTRIBUTION_MAPPING.setdefault("flash_attn", ["flash-attn"])
except Exception:
    pass

# ── Standard imports ──────────────────────────────────────────────────────────
import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("BuildSight.VLM")

# ── Config ───────────────────────────────────────────────────────────────────
# Models tried in order; first successful load wins.
VLM_MODEL_IDS = [
    "florence-community/Florence-2-base-ft",   # fine-tuned (preferred)
    "microsoft/Florence-2-base",                # base model (cached by download_models.py)
]
VLM_CACHE_TTL_S      = 10.0   # seconds between cached re-descriptions
VLM_RETRY_INTERVAL_S = 300.0  # wait this long before retrying after a failed load

# Florence-2 caption task — VQA outputs garbage on this model (returns "yes",
# "0.17", "QA"). MORE_DETAILED_CAPTION generates rich natural-language scene
# descriptions that Turner AI uses as visual grounding context.
VLM_CAPTION_TASK = "<MORE_DETAILED_CAPTION>"

DEFAULT_QUESTION = (
    "Describe the construction site activity. "
    "Are workers wearing helmets and safety vests? "
    "Any obvious safety hazards?"
)

# ── Lazy model state ─────────────────────────────────────────────────────────
_VLM_MODEL: Optional[Any]       = None
_VLM_PROCESSOR: Optional[Any]   = None
_VLM_AVAILABLE: Optional[bool]  = None   # None=never tried, True=OK, False=failed
_VLM_MODEL_ID_LOADED: str       = ""
_VLM_DEVICE: str                = "cpu"
_VLM_DTYPE: Any                 = torch.float32
_VLM_RETRY_AFTER: float         = 0.0    # epoch seconds; 0 = can retry immediately
_VLM_LOAD_LOCK                  = threading.Lock()

# ── Cached result ─────────────────────────────────────────────────────────────
_CACHED_ENTRY: Dict  = {}
_CACHED_TS: float    = 0.0


# ── Model loader ──────────────────────────────────────────────────────────────

def _try_load_vlm() -> bool:
    """
    Try to load Florence-2 (fine-tuned then base). Thread-safe.
    Returns True if a model is loaded and ready.
    Respects the retry backoff so one transient failure doesn't lock the VLM forever.
    """
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_AVAILABLE
    global _VLM_MODEL_ID_LOADED, _VLM_DEVICE, _VLM_DTYPE, _VLM_RETRY_AFTER

    # Already loaded successfully — fast path
    if _VLM_AVAILABLE is True:
        return True

    # Failed previously — check backoff timer
    if _VLM_AVAILABLE is False and time.time() < _VLM_RETRY_AFTER:
        return False

    with _VLM_LOAD_LOCK:
        # Re-check under lock (another thread may have loaded it)
        if _VLM_AVAILABLE is True:
            return True
        if _VLM_AVAILABLE is False and time.time() < _VLM_RETRY_AFTER:
            return False

        # Determine device explicitly — avoids requiring `accelerate` for device_map="auto"
        _target_device = "cuda" if torch.cuda.is_available() else "cpu"
        _load_dtype    = torch.float16 if _target_device == "cuda" else torch.float32

        for model_id in VLM_MODEL_IDS:
            log.info("VLM: Attempting to load %s on %s ...", model_id, _target_device)
            try:
                # Newer transformers has Florence-2 natively — use dedicated classes.
                # AutoModelForCausalLM rejects Florence2Config in recent versions.
                try:
                    from transformers import Florence2ForConditionalGeneration, Florence2Processor
                    processor = Florence2Processor.from_pretrained(
                        model_id, trust_remote_code=True,
                    )
                    model = Florence2ForConditionalGeneration.from_pretrained(
                        model_id,
                        torch_dtype=_load_dtype,
                        trust_remote_code=True,
                    )
                except (ImportError, AttributeError):
                    # Older transformers — fall back to Auto classes
                    from transformers import AutoModelForCausalLM, AutoProcessor
                    processor = AutoProcessor.from_pretrained(
                        model_id, trust_remote_code=True,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=_load_dtype,
                        trust_remote_code=True,
                    )

                # Patch missing forced_bos_token_id on older Florence-2 model configs
                for cfg_attr in ("text_config", "language_config"):
                    lang_cfg = getattr(model.config, cfg_attr, None)
                    if lang_cfg is not None and not hasattr(lang_cfg, "forced_bos_token_id"):
                        lang_cfg.forced_bos_token_id = None

                model = model.to(_target_device)
                model.eval()

                device_str = str(_target_device)
                dtype      = _load_dtype

                if "cuda" in device_str:
                    free, total = torch.cuda.mem_get_info()
                    log.info(
                        "VLM: %s loaded on %s (float16) — VRAM free %.2f GB / %.2f GB",
                        model_id, device_str, free / 1024**3, total / 1024**3,
                    )
                else:
                    log.info("VLM: %s loaded on CPU (float32)", model_id)

                _VLM_MODEL          = model
                _VLM_PROCESSOR      = processor
                _VLM_DEVICE         = device_str
                _VLM_DTYPE          = dtype
                _VLM_MODEL_ID_LOADED = model_id
                _VLM_AVAILABLE      = True
                _VLM_RETRY_AFTER    = 0.0
                log.info("VLM: Ready — model=%s device=%s", model_id, device_str)
                return True

            except Exception as exc:
                log.warning("VLM: Failed to load %s: %s", model_id, exc)
                continue

        # All candidates failed
        log.error("VLM: All model candidates failed. Rule-based fallback active.")
        _VLM_AVAILABLE   = False
        _VLM_RETRY_AFTER = time.time() + VLM_RETRY_INTERVAL_S
        return False


def trigger_preload() -> None:
    """
    Fire VLM loading in a background daemon thread so the server startup
    doesn't block and the model is warm before the first user request.
    """
    if _VLM_AVAILABLE is True:
        return
    t = threading.Thread(target=_try_load_vlm, daemon=True, name="vlm-preload")
    t.start()
    log.info("VLM: Background preload started.")


# ── Inference ─────────────────────────────────────────────────────────────────

def describe_frame_sync(
    frame_bgr: Optional[np.ndarray] = None,
    jpeg_bytes: Optional[bytes]     = None,
    question: str                   = DEFAULT_QUESTION,
    point_prompt: Optional[Tuple[int, int]] = None,
    fallback_stats: Optional[Dict]  = None,
    force_refresh: bool             = False,
) -> Dict:
    """
    Describe a frame using Florence-2, or fall back to rule-based text.

    If no image is provided (frame_bgr and jpeg_bytes both None) but the VLM
    is loaded, returns a rule-based description so the source stays honest.
    """
    global _CACHED_ENTRY, _CACHED_TS
    now = time.time()

    # Cache hit
    if not point_prompt and not force_refresh and _CACHED_ENTRY and (now - _CACHED_TS) < VLM_CACHE_TTL_S:
        return _CACHED_ENTRY

    description = ""
    source      = "rule_based"

    # ── Florence-2 Inference ─────────────────────────────────────────────────
    if _try_load_vlm() and (frame_bgr is not None or jpeg_bytes is not None):
        try:
            from PIL import Image
            import io

            if jpeg_bytes:
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            else:
                import cv2
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

            # Spatial zoom for point prompts
            if point_prompt:
                px, py = point_prompt
                w, h   = img.size
                cs     = 400
                left   = max(0, px - cs // 2)
                top    = max(0, py - cs // 2)
                right  = min(w, left + cs)
                bottom = min(h, top + cs)
                if right == w:  left  = max(0, w - cs)
                if bottom == h: top   = max(0, h - cs)
                img = img.crop((left, top, right, bottom))
                if "focus" not in question.lower():
                    question = f"What is at the center of this focused view? {question}"

            # Use MORE_DETAILED_CAPTION — VQA task returns garbage ("yes", "0.17",
            # "QA") on Florence-2-base-ft. Caption produces rich natural-language
            # scene descriptions that Turner uses as visual grounding context.
            task   = VLM_CAPTION_TASK
            prompt = task
            inputs = _VLM_PROCESSOR(text=prompt, images=img, return_tensors="pt")

            # Move all tensors to model device with correct dtype
            model_device = next(_VLM_MODEL.parameters()).device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model_device, dtype=_VLM_DTYPE) if v.is_floating_point() else v.to(model_device)

            with torch.inference_mode():
                generated_ids = _VLM_MODEL.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True,
                )

            generated_text = _VLM_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed         = _VLM_PROCESSOR.post_process_generation(
                generated_text, task=task, image_size=(img.width, img.height)
            )

            description = str(parsed.get(task, "")).strip()

            # Reject empty or trivially short outputs
            if not description or len(description) < 10:
                description = ""

            source = "florence2"

            if "cuda" in _VLM_DEVICE:
                torch.cuda.empty_cache()

        except Exception as exc:
            log.error("VLM: Inference error — %s", exc)
            description = ""

    # ── Rule-based fallback ───────────────────────────────────────────────────
    if not description:
        description = _rule_based_description(fallback_stats or {}, question)
        source      = "rule_based"

    entry = {
        "description": description,
        "timestamp":   now,
        "source":      source,
        "question":    question,
        "point":       point_prompt,
    }

    if not point_prompt:
        _CACHED_ENTRY = entry
        _CACHED_TS    = now

    return entry


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _scene_condition_summary(scene: str) -> str:
    if "S4" in scene: return "Crowded site conditions with elevated worker density."
    if "S3" in scene: return "Low-light site conditions with reduced visibility."
    if "S2" in scene: return "Dusty site conditions with reduced visibility."
    return "Normal site conditions."


def _rule_based_description(stats: Dict, question: str = DEFAULT_QUESTION) -> str:
    """Generate a live site status summary from detection counts."""
    workers    = stats.get("total_workers", 0) or stats.get("active_workers", 0)
    helmets    = stats.get("helmets_detected", 0)
    vests      = stats.get("vests_detected", 0)
    violations = stats.get("proximity_violations", 0)
    scene      = stats.get("scene", "S1_normal")
    avg_risk   = stats.get("avg_site_risk", 0.0)
    ppe_detail  = stats.get("ppe_violation_detail", "")
    zone_detail = stats.get("zone_breach_detail", "")
    q          = (question or "").strip().lower()
    scene_summary = _scene_condition_summary(scene)

    if not stats or (workers == 0 and helmets == 0 and vests == 0):
        if "happening" in q or "going on" in q or "what" in q or "activity" in q:
            return "Site activity is being monitored. No detections recorded in current session."
        if "safe" in q or "risk" in q or "hazard" in q:
            return "Site appears stable. Awaiting detection data for risk assessment."
        return "Site monitoring active. No worker detections in current frame."

    if "condition" in q or "scene" in q or "visibility" in q or "weather" in q:
        if workers > 0:
            return f"{scene_summary} {workers} worker{'s' if workers != 1 else ''} currently active."
        return scene_summary

    if workers == 0:
        if "hazard" in q or "risk" in q or "safe" in q or "unsafe" in q:
            return f"No workers currently detected on site. {scene_summary}"
        return f"{scene_summary} No workers currently detected on site."

    helmet_pct = int(helmets / max(workers, 1) * 100)
    vest_pct   = int(vests   / max(workers, 1) * 100)
    parts = [f"{workers} worker{'s' if workers != 1 else ''} active on site."]

    if helmet_pct == 100:
        parts.append("All workers wearing helmets.")
    elif helmet_pct >= 80:
        parts.append(f"Helmet compliance: {helmet_pct}%.")
    else:
        missing = workers - helmets
        parts.append(f"{missing} worker{'s' if missing != 1 else ''} missing helmet — {helmet_pct}% compliance.")

    if vest_pct == 100:
        parts.append("All workers wearing high-vis vests.")
    elif vest_pct < 60:
        missing_v = workers - vests
        parts.append(f"{missing_v} worker{'s' if missing_v != 1 else ''} missing vest — {vest_pct}% compliance.")
    else:
        parts.append(f"Vest compliance: {vest_pct}%.")

    if ppe_detail:   parts.append(ppe_detail + " flagged.")
    if zone_detail:  parts.append(zone_detail + " detected — immediate action required.")
    elif violations: parts.append(f"{violations} zone violation{'s' if violations != 1 else ''} active.")

    if "S4" in scene: parts.append("Crowded site — elevated density monitoring active.")
    elif "S3" in scene: parts.append("Low-light conditions — visibility may be impaired.")
    elif "S2" in scene: parts.append("Dusty conditions — PPE verification confidence reduced.")

    if avg_risk > 0.7:   parts.append("Overall site risk: HIGH.")
    elif avg_risk > 0.4: parts.append("Moderate overall risk level.")

    summary = " ".join(parts)

    if "happening" in q or "activity" in q or "going on" in q:
        return f"{scene_summary} {summary}"
    if "helmet" in q or "vest" in q or "ppe" in q or "wearing" in q or "compliance" in q:
        return f"{summary} {scene_summary}"
    if "hazard" in q or "risk" in q or "safe" in q or "unsafe" in q or "danger" in q:
        risk_note = f" {zone_detail} detected." if zone_detail else (
            f" {violations} active zone violation{'s' if violations != 1 else ''} detected." if violations else ""
        )
        return f"{scene_summary} {summary}{risk_note}"

    return f"{scene_summary} {summary}"


# ── Public helpers ─────────────────────────────────────────────────────────────

def get_cached_entry() -> Tuple[Dict, bool]:
    """Return (entry_dict, is_stale)."""
    stale = not _CACHED_ENTRY or (time.time() - _CACHED_TS) > VLM_CACHE_TTL_S
    return _CACHED_ENTRY, stale


def is_available() -> bool:
    """Return True if a VLM model is loaded and ready."""
    return _VLM_AVAILABLE is True


def get_model_info() -> Dict:
    """Return diagnostic info about the loaded VLM."""
    return {
        "model_id":    _VLM_MODEL_ID_LOADED or "(not loaded)",
        "candidates":  VLM_MODEL_IDS,
        "available":   _VLM_AVAILABLE is True,
        "device":      _VLM_DEVICE,
        "dtype":       str(_VLM_DTYPE),
        "cache_ttl_s": VLM_CACHE_TTL_S,
        "retry_in_s":  max(0.0, _VLM_RETRY_AFTER - time.time()) if _VLM_AVAILABLE is False else 0.0,
    }


# ── Async wrapper ─────────────────────────────────────────────────────────────

async def describe_frame_async(
    frame_bgr: Optional[np.ndarray]         = None,
    jpeg_bytes: Optional[bytes]             = None,
    question: str                           = DEFAULT_QUESTION,
    point_prompt: Optional[Tuple[int, int]] = None,
    fallback_stats: Optional[Dict]          = None,
    force_refresh: bool                     = False,
) -> Dict:
    """Async wrapper — offloads blocking VLM inference to the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: describe_frame_sync(
            frame_bgr, jpeg_bytes, question, point_prompt, fallback_stats, force_refresh
        ),
    )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 60)
    print("  BuildSight VLM — Florence-2 Isolation Test")
    print("=" * 60)

    print(f"\n[1] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"    Free VRAM: {free / 1024**3:.2f} GB / Total: {total / 1024**3:.2f} GB")

    print("\n[2] Loading VLM (with fallback)...")
    ok = _try_load_vlm()
    print(f"    Loaded: {ok}")
    print(f"    Model:  {_VLM_MODEL_ID_LOADED}")
    print(f"    Device: {_VLM_DEVICE} / Dtype: {_VLM_DTYPE}")

    print("\n[3] Rule-based fallback test...")
    result = describe_frame_sync(fallback_stats={"total_workers": 3, "helmets_detected": 2, "vests_detected": 3})
    print(f"    Source: {result['source']}")
    print(f"    Text:   {result['description'][:100]}...")

    if ok:
        print("\n[4] VLM inference with synthetic frame...")
        dummy  = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = describe_frame_sync(frame_bgr=dummy, force_refresh=True)
        print(f"    Source: {result['source']}")
        print(f"    Text:   {result['description'][:150]}...")
        if torch.cuda.is_available():
            print(f"    VRAM after inference: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    print("\n[OK] VLM isolation test complete.")
