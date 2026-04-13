# Backup Notes — Post-Auto-Mode FPS / Overlap / Clutter Fix
**Timestamp:** 2026-04-11  
**Purpose:** Pre-modification snapshot of the pipeline state after the auto scene
classification update (2026-04-11_auto_scene_routing) and before the FPS recovery,
duplicate-box suppression, and static clutter hardening pass.

---

## Files Backed Up

| File | Source Path |
|------|-------------|
| server.py | dashboard/backend/server.py |
| site_aware_ensemble.py | scripts/site_aware_ensemble.py |
| adaptive_postprocess.py | scripts/adaptive_postprocess.py |
| ensemble_video.py | scripts/ensemble_video.py |
| gemini_auditor.py | scripts/gemini_auditor.py |
| DetectionPanel.tsx | dashboard/src/components/DetectionPanel.tsx |
| DetectionPanel.css | dashboard/src/components/DetectionPanel.css |

---

## Current Measured Performance (Post-Auto-Mode, Pre-Fix)

| Metric | S1_normal | S4_crowded |
|--------|-----------|------------|
| Typical end-to-end latency | ~160–200ms | ~350–500ms |
| Effective FPS | ~5–6 | ~2–3 |
| Full-frame inference (both models) | ~80–100ms | ~80–100ms |
| Tile inference (4×2 model calls) | N/A | ~250–350ms (runs every frame!) |
| Scene classification | ~5ms (cached stats) | ~30–50ms (model forward pass) |
| Post-processing | ~10–15ms | ~10–15ms |

**Root cause of lag:** Auto-mode calls `classify_scene_fast()` every single frame,
including a full YOLOv11 model forward pass. In S4_crowded, tile inference
(`_infer_tiles`) runs 4 additional predict() calls every frame — 6 total model
calls per frame on the GPU instead of the normal 2.

---

## Current Scene Classification Logic

- `classify_scene_fast()` runs every frame unconditionally
- Checks brightness/contrast/saturation from image statistics (~1ms)
- Runs a full `model_v11.predict()` pass at conf=0.22 for worker count
- Result fed through `SceneConditionTracker.update()` for hysteresis
- **No caching** — model forward pass on every single frame

---

## Current Crowded-Scene (S4) Thresholds

### server.py — WBF IoU (S4_crowded)
```python
wbf_iou_cond = {0: 0.45, 1: 0.50, 2: 0.72}  # worker IoU=0.72
```
Worker WBF IoU = 0.72 means two boxes must overlap ≥ 72% to be fused.
Two nearby workers (IoU ~0.3–0.55) are intentionally NOT merged.
**Problem:** Two outputs from the same worker by two models (IoU ~0.60–0.70) are
also not being merged, creating duplicate boxes for the same person.

### server.py — POST_WBF_BY_CONDITION (S4_crowded)
```python
"S4_crowded": {0: 0.28, 1: 0.12, 2: 0.15}
```
Worker post-gate = 0.15 — very permissive, allows low-confidence detections
from materials and clutter to pass.

### server.py — is_valid_worker() (S4_crowded)
- Height floor: bh < img_h * 0.045 → rejected (lowered to catch wall workers)
- Max aspect: 2.4
- Portrait check: aspect > 1.80 AND score < 0.58 → rejected (very relaxed)

---

## Current Overlap and NMS Suppression Rules

### server.py — wbf_fuse()
- Worker WBF IoU: 0.65 (S1_normal), 0.72 (S4_crowded)
- Pre-WBF confidence: 0.15 (S4), 0.20 (S1)
- Post-WBF gate: worker 0.15–0.18 across conditions

### server.py — _merge_full_and_tile()
- Deduplication IoU: **0.35** — two tile boxes vs full-frame boxes merge if IoU ≥ 0.35
- This is too low: same-person boxes from tile pass that overlap 40–60% are NOT deduped

### site_aware_ensemble.py — recover_crowded_workers()
- New candidate suppressed if IoU ≥ **0.55** with existing workers
- Problem: two boxes for the same person can have IoU 0.55–0.65, both surviving

### adaptive_postprocess.py — cross_class_nms()
- Same-class NMS IoU: **0.45**
- Applied before the association step

---

## Current False-Positive Categories Observed

After the auto-mode + recall update, the following are being generated:

1. **Cement bags / sand bags** — wide, flat geometry; survive lower S4 aspect filter
2. **Blue plastic buckets** — short, rounded; height filter relaxed too much
3. **Scaffolding cross-members** — high edge density; survive even with hard negative suppressor
4. **Wooden planks / poles** — elongated, low confidence (score 0.22–0.38)
5. **Stacked bricks / rebar clusters** — small boxes in corners, survive size floor
6. **Material piles / tarpaulins** — large landscape boxes, wide aspect passes S4 filter
7. **Machinery parts / formwork** — irregular geometry, some survive portrait check

---

## Current Frontend Auto-Mode Behavior

- Default mode: **AUTO** (`autoMode=true` state)
- Sends `condition: "auto"` form field to `/api/detect/frame`
- Receives `data.condition` back from server and updates `detectedCondition` state
- SceneAutoIndicator shows live detected condition badge with color coding
- `liveFps` = 5-frame rolling average of `1000 / frameElapsed`
- MIN_GAP_MS = 80ms between inference calls (inference bottleneck dominates)
- `isFirstFrameRef.current = true` on new file load → sends `reset_tracker=1` once

---

## Known Regressions After Auto-Mode Update

1. **FPS dropped from ~6–8 to ~2–3 in S4_crowded** — tile inference runs every frame
2. **Scene classifier model forward pass every frame** — 30–50ms extra overhead regardless of scene
3. **Duplicate boxes (3–4 per worker)** — WBF IoU=0.72 doesn't merge same-person model disagreements; tile recovery adds more
4. **Clutter FPs increased** — post-gate lowered to 0.15 in S4, aspect filter relaxed to 2.4, height floor 0.045 allows tiny blobs
5. **Worker recovery re-creates boxes over tracked workers** — IoU=0.55 check in recover_crowded_workers too low
6. **No final post-merge NMS** — after tiling and recovery add boxes, there is no global deduplication pass

---

## Active Model Weights (Unchanged)
- YOLOv11: `weights/yolov11_buildsight_best.pt`
- YOLOv26: `weights/yolov26_buildsight_best.pt`
- WBF model weights: [0.55 (v11), 0.45 (v26)]
- Device: CUDA:0 (RTX 4050), FP16 enabled
