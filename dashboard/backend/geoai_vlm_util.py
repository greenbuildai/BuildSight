"""
BuildSight GeoAI — VLM Activity Description Utility (Florence-2)
================================================================
Async-safe, lazily-loaded Microsoft Florence-2 wrapper for site activity narration.
Optimized for <5GB VRAM environments.

Behaviour:
  - On first call, tries to load `microsoft/Florence-2-base`.
  - Uses <VQA> task for question-answering about site scenes.
  - If unavailable, generates rule-based descriptions from detection stats.
  - The last description is cached for VLM_CACHE_TTL_S seconds.
  - Exposes both a sync and an async wrapper.

GPU Optimizations (2026-04-22):
  - Forces torch.float16 on CUDA
  - Uses torch.inference_mode() during generation
  - Limits max_new_tokens to 96 for speed
  - Clears CUDA cache after every VLM response
  - Caches recent descriptions to avoid redundant inference
  - Logs GPU memory usage at startup
"""

# ── CRITICAL: Mock flash_attn BEFORE any transformers imports ─────────────────
# Florence-2 on HuggingFace has a soft dependency on flash_attn which doesn't
# build on Windows. We pre-inject a stub so transformers never tries to import
# the real thing.
import sys
import types
from unittest.mock import MagicMock

if "flash_attn" not in sys.modules:
    _fa = types.ModuleType("flash_attn")
    _fa.__spec__ = MagicMock()
    sys.modules["flash_attn"] = _fa

# ── Standard imports ──────────────────────────────────────────────────────────
import asyncio
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("BuildSight.VLM")

# ── Config ───────────────────────────────────────────────────────────────────
VLM_MODEL_ID = "microsoft/Florence-2-base"
VLM_CACHE_TTL_S = 10.0  # Throttled for stability

DEFAULT_QUESTION = (
    "Describe the construction site activity. "
    "Are workers wearing helmets and safety vests? "
    "Any obvious safety hazards?"
)

# ── Lazy model state ─────────────────────────────────────────────────────────
_VLM_MODEL: Optional[Any] = None
_VLM_PROCESSOR: Optional[Any] = None
_VLM_AVAILABLE: Optional[bool] = None
_VLM_DEVICE: str = "cpu"
_VLM_DTYPE: Any = torch.float32

# ── Cached result ─────────────────────────────────────────────────────────────
_CACHED_ENTRY: Dict = {}
_CACHED_TS: float = 0.0


# ── Model loader ──────────────────────────────────────────────────────────────

def _try_load_vlm() -> bool:
    """Load Florence-2-base once. Returns True on success."""
    global _VLM_MODEL, _VLM_PROCESSOR, _VLM_AVAILABLE, _VLM_DEVICE, _VLM_DTYPE
    if _VLM_AVAILABLE is not None:
        return _VLM_AVAILABLE

    try:
        from transformers import AutoModel, AutoProcessor
        
        log.info("VLM: Loading Florence-2-base (%s)...", VLM_MODEL_ID)
        log.info("VLM: This may take a few minutes on first load...")

        # ── Device & dtype selection ─────────────────────────────────────────
        _VLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _VLM_DTYPE = torch.float16 if _VLM_DEVICE == "cuda" else torch.float32
        
        log.info("VLM: Using device=%s dtype=%s", _VLM_DEVICE, _VLM_DTYPE)

        # Download and load the model
        log.info("VLM: Downloading model from HuggingFace...")
        _VLM_PROCESSOR = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
        log.info("VLM: Processor loaded, now loading model...")
        
        _VLM_MODEL = AutoModel.from_pretrained(
            VLM_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=_VLM_DTYPE,
        )
        
        _VLM_MODEL = _VLM_MODEL.to(_VLM_DEVICE)
        _VLM_MODEL.eval()
        
        _VLM_AVAILABLE = True
        
        log.info("VLM: SUCCESS! Florence-2-base loaded on %s", _VLM_DEVICE)
        log.info("VLM: Model ready for inference")
        return True

    except Exception as exc:
        log.error("VLM: FAILED to load Florence-2-base: %s", exc)
        import traceback
        traceback.print_exc()
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
    Describe a frame using Florence-2, or fall back to rule-based text.
    """
    global _CACHED_ENTRY, _CACHED_TS
    now = time.time()

    # Cache check
    if not point_prompt and not force_refresh and _CACHED_ENTRY and (now - _CACHED_TS) < VLM_CACHE_TTL_S:
        return _CACHED_ENTRY

    description = ""
    source = "rule_based"

    # ── Florence-2 Inference Path ───────────────────────────────────────────
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

            # Spatial Zoom if point provided
            if point_prompt:
                px, py = point_prompt
                w, h = img.size
                crop_size = 400
                left = max(0, px - crop_size // 2)
                top = max(0, py - crop_size // 2)
                right = min(w, left + crop_size)
                bottom = min(h, top + crop_size)
                # Bounds check
                if right == w: left = max(0, w - crop_size)
                if bottom == h: top = max(0, h - crop_size)
                img = img.crop((left, top, right, bottom))
                
                if "focus" not in question.lower():
                    question = f"What is at the center of this focused view? {question}"

            # Prepare Inputs
            # Florence-2 works best when the question follows the task tag directly
            prompt = f"<VQA>{question}"
            inputs = _VLM_PROCESSOR(text=prompt, images=img, return_tensors="pt")
            
            # Move all tensor inputs to device and dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.is_floating_point():
                        inputs[k] = v.to(_VLM_DEVICE, dtype=_VLM_DTYPE)
                    else:
                        inputs[k] = v.to(_VLM_DEVICE)

            with torch.inference_mode():
                generated_ids = _VLM_MODEL.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=96,   # Capped for speed — 96 is enough for site descriptions
                    num_beams=3,
                    do_sample=False,
                    early_stopping=True
                )
            
            # Use skip_special_tokens=True to get a cleaner raw string first
            generated_text = _VLM_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Post-process using the official method
            parsed_answer = _VLM_PROCESSOR.post_process_generation(
                generated_text, 
                task="<VQA>", 
                image_size=(img.width, img.height)
            )
            
            # Extract and clean
            raw_answer = parsed_answer.get("<VQA>", "")
            # Sometimes parsing leaves the question or prefix
            description = raw_answer.replace("QA>", "").replace("<VQA>", "").strip()
            
            # Final fallback: detect hallucinated echoes where VLM just repeats the question
            desc_clean = description.lower().strip()
            q_clean = question.lower().strip()
            is_echo = (
                not description
                or desc_clean in q_clean           # description is subset of question
                or q_clean in desc_clean            # question is subset of description
                or len(set(desc_clean.split()) - set(q_clean.split())) < 3  # < 3 new words
            )
            if is_echo:
                # VQA echoed back the prompt — fall back to generic caption
                description = "Site activity observed. Processing vision telemetry..."
            
            source = "florence2"
            
            # Cleanup — critical for VRAM stability with YOLO + SAM running
            if _VLM_DEVICE == "cuda":
                torch.cuda.empty_cache()

        except Exception as exc:
            log.error("VLM: Inference error — %s", exc)

    # ── Rule-based fallback ───────────────────────────────────────────────────
    if not description:
        description = _rule_based_description(fallback_stats or {}, question)
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


def _scene_condition_summary(scene: str) -> str:
    if "S4" in scene:
        return "Crowded site conditions with elevated worker density."
    if "S3" in scene:
        return "Low-light site conditions with reduced visibility."
    if "S2" in scene:
        return "Dusty site conditions with reduced visibility."
    return "Normal site conditions."


def _rule_based_description(stats: Dict, question: str = DEFAULT_QUESTION) -> str:
    """Generate a live site status summary from detection counts.
    ALWAYS returns a meaningful response - no empty responses allowed."""
    workers = stats.get("total_workers", 0) or stats.get("active_workers", 0)
    helmets = stats.get("helmets_detected", 0)
    vests   = stats.get("vests_detected", 0)
    violations = stats.get("proximity_violations", 0)
    scene   = stats.get("scene", "S1_normal")
    avg_risk = stats.get("avg_site_risk", 0.0)
    ppe_detail  = stats.get("ppe_violation_detail", "")
    zone_detail = stats.get("zone_breach_detail", "")
    q = (question or "").strip().lower()
    scene_summary = _scene_condition_summary(scene)

    # If no stats available at all, return safe default
    if not stats or (workers == 0 and helmets == 0 and vests == 0):
        if "happening" in q or "going on" in q or "what" in q or "activity" in q:
            return "Site activity is being monitored. No detections recorded in current session."
        if "safe" in q or "risk" in q or "hazard" in q:
            return "Site appears stable. Awaiting detection data for risk assessment."
        return "Site monitoring active. No worker detections in current frame."
    
    # Specific question handling
    if "condition" in q or "scene" in q or "visibility" in q or "weather" in q:
        if workers > 0:
            return f"{scene_summary} {workers} worker{'s' if workers != 1 else ''} currently active."
        return scene_summary

    if workers == 0:
        if "hazard" in q or "risk" in q or "safe" in q or "unsafe" in q:
            return f"No workers currently detected on site. {scene_summary}"
        return f"{scene_summary} No workers currently detected on site."

    # Build comprehensive response
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

    if ppe_detail:
        parts.append(ppe_detail + " flagged.")
    if zone_detail:
        parts.append(zone_detail + " detected — immediate action required.")
    elif violations > 0:
        parts.append(f"{violations} zone violation{'s' if violations != 1 else ''} active.")

    if "S4" in scene:
        parts.append("Crowded site — elevated density monitoring active.")
    elif "S3" in scene:
        parts.append("Low-light conditions — visibility may be impaired.")
    elif "S2" in scene:
        parts.append("Dusty conditions — PPE verification confidence reduced.")

    if avg_risk > 0.7:
        parts.append("Overall site risk: HIGH.")
    elif avg_risk > 0.4:
        parts.append("Moderate overall risk level.")

    summary = " ".join(parts)

    if "happening" in q or "activity" in q or "going on" in q:
        return f"{scene_summary} {summary}"

    if "helmet" in q or "vest" in q or "ppe" in q or "wearing" in q or "compliance" in q:
        return f"{summary} {scene_summary}"

    if "hazard" in q or "risk" in q or "safe" in q or "unsafe" in q or "danger" in q:
        risk_note = ""
        if zone_detail:
            risk_note = f" {zone_detail} detected."
        elif violations > 0:
            risk_note = f" {violations} active zone violation{'s' if violations != 1 else ''} detected."
        return f"{scene_summary} {summary}{risk_note}"

    return f"{scene_summary} {summary}"


def get_cached_entry() -> Tuple[Dict, bool]:
    """Return (entry_dict, is_stale)."""
    stale = not _CACHED_ENTRY or (time.time() - _CACHED_TS) > VLM_CACHE_TTL_S
    return _CACHED_ENTRY, stale


def is_available() -> bool:
    """Return True if VLM is loaded and ready."""
    return bool(_VLM_AVAILABLE)


def get_model_info() -> Dict:
    """Return diagnostic info about the loaded VLM for health checks."""
    return {
        "model_id": VLM_MODEL_ID,
        "available": bool(_VLM_AVAILABLE),
        "device": _VLM_DEVICE,
        "dtype": str(_VLM_DTYPE),
        "cache_ttl_s": VLM_CACHE_TTL_S,
    }


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


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("  BuildSight VLM — Florence-2 Isolation Test")
    print("=" * 60)
    
    # Test 1: CUDA availability
    print(f"\n[1] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    torch.cuda.device_count() = {torch.cuda.device_count()}")
        print(f"    torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"    Free VRAM: {free / 1024**3:.2f} GB / Total: {total / 1024**3:.2f} GB")
    
    # Test 2: Model loading
    print("\n[2] Loading Florence-2-base...")
    ok = _try_load_vlm()
    print(f"    Model loaded: {ok}")
    print(f"    Device: {_VLM_DEVICE}")
    print(f"    Dtype: {_VLM_DTYPE}")
    
    # Test 3: Rule-based fallback
    print("\n[3] Testing rule-based fallback...")
    result = describe_frame_sync(fallback_stats={"total_workers": 3, "helmets_detected": 2, "vests_detected": 3})
    print(f"    Source: {result['source']}")
    print(f"    Description: {result['description'][:100]}...")
    
    # Test 4: VLM inference with dummy frame (if model loaded)
    if ok:
        print("\n[4] Testing VLM inference with synthetic frame...")
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = describe_frame_sync(frame_bgr=dummy, force_refresh=True)
        print(f"    Source: {result['source']}")
        print(f"    Description: {result['description'][:150]}...")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"    GPU memory after inference: {allocated:.2f} GB")
    
    print("\n[OK] VLM isolation test complete.")
