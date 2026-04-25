# theo_handoff.md — Theo's Task File (OpenCode Executor)

> **Role**: Tertiary Executor — Optimization, Refactoring, Validation
> **Priority**: 3rd in line (after Toni → Leon)
> **Activation**: Theo activates when Toni and Leon are both unavailable, OR when a task specifically requires optimization/profiling/code quality work.

---

## Theo's Responsibilities

- Code optimization and performance profiling
- Refactoring messy or duplicated logic
- Validating output from Toni and Leon
- Running static analysis and catching edge cases
- Handling full execution when escalated

---

## How Theo Reads This File

1. Check **Active Tasks** below — if your name is listed or status is `pending`, start immediately
2. Read the full objective, constraints, and expected output before touching any file
3. Mark task `in_progress` when you begin
4. Mark `completed` and write a brief result note when done
5. If blocked, add a `blocked` note and escalate to Jovi

---

## Active Tasks

### Task #001 — Florence-2 VLM Full Fix
- **Priority**: High
- **Status**: `completed`
- **Assigned To**: Theo (OpenCode)
- **File**: `dashboard/backend/geoai_vlm_util.py`
- **Result**: Changed to `florence-community/Florence-2-base-ft` with `AutoModelForCausalLM` and `device_map="auto"`. Inference device now uses model's actual device.

---

**Objective**: The VLM is running in `rule_based` fallback mode — Florence-2 is never actually loading or running inference. Fix the model loading, inference pipeline, and all fallback logic completely.

**Current confirmed bug**: API returns `"source": "rule_based"` on every call. No VLM log entries in `server_log.txt`. The model class `AutoModel` cannot attach Florence-2's generation head — it silently fails.

---

## 🔹 1. Model Loading Fix

Replace the import and model loader in `geoai_vlm_util.py`:

```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

VLM_MODEL_ID = "florence-community/Florence-2-base-ft"

processor = AutoProcessor.from_pretrained(
    VLM_MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

# Add this for GPU debug confirmation
print(f"[VLM] Loaded on device: {model.device}")
```

> ⚠️ **Important**: If using `device_map="auto"`, remove the `.to(_VLM_DEVICE)` call on line 100.
> These two cannot be used together — `device_map="auto"` handles placement automatically.

---

## 🔹 2. Inference Code (Replace existing inference block)

Replace the inference function with this complete, corrected version:

```python
def run_vlm_inference(image, user_query, telemetry=None):

    # ---- Safety check ----
    if image is None:
        return "Live scene frame not available. Using telemetry only."

    # ---- Build context-aware prompt ----
    context = ""
    if telemetry:
        workers   = telemetry.get('workers', 0)
        violations = telemetry.get('violations', 0)
        zone      = telemetry.get('critical_zone', 'unknown')
        helmet    = telemetry.get('helmet_missing', False)
        vest      = telemetry.get('vest_missing', False)

        context = f"""
Workers: {workers}
Violations: {violations}
Critical Zone: {zone}
Missing PPE: helmet={helmet}, vest={vest}
Risk: {telemetry.get('risk_level', 'unknown')}
"""

    prompt = f"<VQA>\nQuestion: {user_query}\nScene Context: {context}"

    # ---- Prepare inputs ----
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ---- Run inference safely ----
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=2,
            do_sample=False
        )

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # ---- Fallback if model output is empty ----
    if not output or len(output.strip()) == 0:
        return generate_rule_based_summary(telemetry)

    # ---- GPU cleanup ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output
```

---

## 🔹 3. Rule-Based Fallback (Always returns something)

Add/replace `_rule_based_description` with this version that never returns empty:

```python
def generate_rule_based_summary(telemetry):
    if not telemetry:
        return "No telemetry available. System initializing."

    workers    = telemetry.get('workers', 0)
    violations = telemetry.get('violations', 0)
    risk       = telemetry.get('risk_level', 'low')

    return (
        f"{workers} workers detected. "
        f"{violations} PPE violations. "
        f"Risk level: {risk}."
    )
```

---

## 🔹 4. Critical Bug Fixes

### Fix 1 — "Awaiting scene frame…" stuck state
- Ensure the latest detection frame is always passed into the VLM call
- If frame is `None`, return the fallback immediately:
```python
if frame is None:
    return fallback_response
```

### Fix 2 — VLM not responding to "Ask" button
- Confirm `/vlm/query` endpoint is being called when user submits a question
- Check browser DevTools → Network tab → look for the POST request
- If request is missing, the frontend fetch() is silently failing

### Fix 3 — GPU throttling (DO NOT run VLM every frame)
- ❌ Don't call VLM on every detection frame — this will OOM the GPU
- ✅ Only call VLM:
  - When user explicitly asks a question
  - OR on a 2–3 second interval max

---

## 🔹 5. Verification Steps

After making all changes, restart the backend:
```
python start_backend.py
```

Then check:
1. Logs should show: `[VLM] Loaded on device: cuda:0`
2. Call the API:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/geoai/vlm/latest" | ConvertTo-Json
```
3. Response should show `"source": "florence2"` — NOT `"rule_based"`
4. Ask a question in the GeoAI panel — should respond with scene intelligence

---

## 🔹 6. Next Step After This Works (From Jovi)

After Florence-2 is confirmed working:

> Combine **Florence-2 (vision) + Turner AI (reasoning)**
> This gives supervisor-level answers, risk explanations, and site intelligence narratives.

Theo to flag when Task #001 is complete so Toni can plan the Turner AI integration.

---

## Completed Tasks

*None yet*

---

## Notes from Toni (Context)

- Backend entry: `dashboard/backend/start_backend.py`
- VLM file: `dashboard/backend/geoai_vlm_util.py`
- The `flash_attn` mock at lines 23–34 of `geoai_vlm_util.py` is **critical** — do not remove or move it
- v1.8 Beta is live on GitHub (`tag: v1.8-beta`) — commit changes with a clear message after completion
- API base: `http://localhost:8000`
- Frontend: `http://localhost:5173`

---

*Last updated by: Toni (Claude) — 2026-04-23 | Tasked by: Jovi*
