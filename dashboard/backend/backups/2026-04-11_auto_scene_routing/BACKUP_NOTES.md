# Backup Notes — Auto Scene Routing + Recall + FPS Fix
**Timestamp:** 2026-04-11  
**Purpose:** Pre-modification snapshot before implementing automatic scene classification,
missed-worker recall improvements, and real-time FPS optimization.

---

## Files Backed Up

| File | Source Path |
|------|-------------|
| server.py | dashboard/backend/server.py |
| site_aware_ensemble.py | scripts/site_aware_ensemble.py |
| adaptive_postprocess.py | scripts/adaptive_postprocess.py |
| ensemble_video.py | scripts/ensemble_video.py |
| gemini_auditor.py | scripts/gemini_auditor.py |
| pipeline_config.py | scripts/pipeline_config.py |

---

## Current Scene Classification Rules (Manual — Pre-Fix)

The frontend sends `condition` as a form field to `/api/detect/frame`.  
The user must manually press one of four buttons: S1 Normal, S2 Dusty, S3 Low Light, S4 Crowded.  
**No automatic detection is implemented on the backend.**

### Consequence
- When a site with 4+ workers is loaded, the system defaults to S1 Normal
- S4_crowded thresholds are never applied unless the operator manually switches
- Workers on elevated structures are missed because S1 uses stricter aspect/height filters
- No hysteresis — condition can be changed instantly with no stability check

---

## Current Ensemble Routing Logic

| Condition | pre_conf | WBF worker IoU | post_gate worker | NMS IoU |
|-----------|----------|----------------|------------------|---------|
| S1_normal | 0.20 | 0.65 | 0.18 | 0.60 |
| S2_dusty | 0.17 | 0.65 | 0.15 | 0.60 |
| S3_low_light | 0.15 | 0.60 | 0.14 | 0.60 |
| S4_crowded | 0.15 | 0.72 | 0.15 | 0.65 |

Early-exit: disabled for S4_crowded, enabled otherwise at EARLY_EXIT_CONF=0.75

---

## Current S1/S4 Thresholds

### server.py — POST_WBF_BY_CONDITION
```python
"S1_normal":    {0: 0.30, 1: 0.14, 2: 0.18},
"S2_dusty":     {0: 0.25, 1: 0.10, 2: 0.15},
"S3_low_light": {0: 0.22, 1: 0.10, 2: 0.14},
"S4_crowded":   {0: 0.28, 1: 0.12, 2: 0.15},
```

### server.py — is_valid_worker() height floor
- S1_normal: bh < img_h * 0.08 AND score < 0.60 → rejected
- S4_crowded: bh < img_h * 0.06 AND score < 0.60 → rejected

### server.py — is_valid_worker() aspect ratio
- S1_normal: max_aspect = 2.0
- S4_crowded: max_aspect = 2.4
- Portrait check S1: aspect > 0.88 AND score < 0.70 → rejected

---

## Gemini Auditor Status
- Env: BUILDSIGHT_GEMINI_AUDITOR_ENABLED (default True)
- Role: Validator-only (supplement mode disabled since last fix)
- Trigger: Ambiguous detections 0.30–0.68 conf only
- Daily quota: 20 requests/day (free tier auto-disables when exhausted)

---

## Active Model Weights
- YOLOv11: `weights/yolov11_buildsight_best.pt`
- YOLOv26: `weights/yolov26_buildsight_best.pt`
- WBF model weights: [0.55 (v11), 0.45 (v26)]
- Device: CUDA:0 (RTX 4050), FP16 enabled

---

## Frontend Architecture (Pre-Fix)

### Condition selection
- `ConditionPicker` component renders 4 buttons: S1 / S2 / S3 / S4
- Default state: `useState<Condition>('S1_normal')`
- User must manually select — no auto mode exists

### Inference loop (VideoUploadMode)
- Self-scheduling async inference loop (not `setInterval`)
- MIN_GAP_MS controls minimum time between inferences
- Frames resized to max 640px before sending
- Condition sent as form field `condition`
- Response includes `elapsed_ms` but no `detected_condition`

### FPS display
- `elapsed_ms` is shown as "LATENCY" in the summary card
- No live FPS calculation or display

---

## Known Issues Before Modification

1. **Manual condition = S1 when 4+ workers visible** — S4_crowded never activates automatically
2. **Elevated wall workers missed** — S1 height floor (0.08 × img_h) rejects distant workers
3. **Grouped workers merged** — WBF IoU for workers (0.65) merges adjacent workers in crowded scenes  
4. **No temporal stability** — condition can flicker between S1 and S4 frame-by-frame if manual
5. **No tiling** — no multi-scale or patch-based inference for distant/edge workers
6. **FPS lag** — no frame dropping, no skip strategy, inference blocks canvas rendering
7. **Missing FPS counter** — only latency shown, no calculated FPS
8. **No auto mode indicator** — operator must always remember to set the mode
