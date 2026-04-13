# Backup Notes — S4 Crowded Scene Worker Recall Fix
**Date**: 2026-04-11  
**Purpose**: Before improving S4 crowded-scene worker and PPE detection recall

---

## Current S4 Crowded-Scene Thresholds (Pre-Fix)

### `buildsight_ensemble.py` — S4 Profile
```python
"S4_crowded": {
    "pre_conf":       0.08,
    "wbf_iou":        {CLS_HELMET: 0.38, CLS_VEST: 0.44, CLS_WORKER: 0.40},
    "post_gate":      {CLS_HELMET: 0.18, CLS_VEST: 0.22, CLS_WORKER: 0.20},
    "recover_worker": 0.32,
}
RECOVERY_COOLDOWN_FRAMES = 4
```

### `site_aware_ensemble.py` — S4 Profile
```python
"S4_crowded": {
    "pre_conf": 0.08,
    "wbf_iou": {CLS_HELMET: 0.38, CLS_VEST: 0.44, CLS_WORKER: 0.40},
    "post_gate": {CLS_HELMET: 0.18, CLS_VEST: 0.22, CLS_WORKER: 0.20},
    "recover_worker": 0.32,
}
RECOVERY_COOLDOWN_FRAMES = 4
```

### `adaptive_postprocess.py` — S4 Thresholds
```python
CONF_THRESHOLDS = {
    "S4_crowded": {"worker": 0.42, "helmet": 0.24, "safety_vest": 0.36},
}
WORKER_MIN_PIXEL_HEIGHT = {"S4_crowded": 22}
WORKER_MIN_HUMAN_SCORE  = {"S4_crowded": 0.36}
NMS_IOU                 = {"S4_crowded": 0.35}
```

### `server.py` — S4 Runtime Gates
```python
POST_WBF_BY_CONDITION = {
    "S4_crowded": {0: 0.28, 1: 0.12, 2: 0.20},  # 0=helmet, 1=vest, 2=worker
}
# is_valid_worker for S4:
#   min_h_frac  = 0.045  (worker height floor as fraction of frame height)
#   max_aspect  = 2.4    (max width/height ratio)
#   size_frac   = 0.008  (size floor fraction)
#   single_thresh = 0.40 (single-model threshold)
#   portrait check: aspect > 1.60 and score < 0.62 → reject

THRESHOLD_STATE defaults:
    global_floor: 0.35
    worker:       0.40
    helmet:       0.45
    vest:         0.35

# ValidWorkerValidator
min_height_px = 40
min_human_score = 0.35

# MaterialSuppressionLayer
min_worker_height_px = 40
```

### `_recover_crowded_workers` — Survival Conditions (Pre-Fix)
```python
min_score = 0.32  # candidates from model output must score ≥ 0.32

# A candidate is recovered if:
#   ppe_support > 0                               (PPE inside worker box)
#   OR (neighbor_support > 0 AND score >= 0.52)   (near other workers, high conf)
#   OR score >= 0.56                              (high standalone confidence)

# Duplicate suppression: IoU ≥ 0.65 → skip
# Height floor: bh < 24 → skip
# Aspect filter: bw/bh > 0.90 → skip
```

---

## Current S4 Routing Logic

1. `pre_conf = min(conf_thresh=0.20, 0.15) = 0.15` — both models run at 0.15
2. `wbf_fuse_condition()` — second-pass WBF from raw model outputs, post_gate=0.20 for workers
3. `recover_crowded_workers()` — recovery with min_score=0.32
4. Tile inference — triggered if worker count < 3 after WBF
5. `is_valid_worker()` — geometry + single-model filtering
6. Threshold gate: `THRESHOLD_STATE["worker"] = 0.40`
7. `ValidWorkerValidator` gate A: conf ≥ 0.42
8. `ValidWorkerValidator` gate C: height ≥ 40px

**Key bottleneck**: The chain `recover_worker >= 0.32` → `single_thresh = 0.40` → `final gate = 0.40` → `ValidWorkerValidator min_height=40` → `material_suppressor min_height=40` forms a **multi-stage filter that's too strict for S4 elevated/distant/occluded workers**.

---

## Current False-Negative Issues in Crowded Scenes

- Workers standing on elevated walls (~2m above ground): smaller bounding boxes, filtered by pixel height floors
- Workers in top-left / top-center of frame: distant, low confidence, below min thresholds
- Workers partially hidden behind scaffolding: low score from models, below recovery threshold
- Workers grouped closely: WBF worker IoU=0.40 may merge two real workers into one box
- Workers with no visible PPE: harder to recover, survival conditions not met
- First-frame workers: ghost suppression `score < 0.72` filters new workers on frame 1

---

## Known FPS and Latency (Pre-Fix)

- Frame inference: ~150-280ms per frame (GPU, RTX 4050)
- Tile inference adds ~120ms when triggered (4 tiles × ~30ms each)
- Gemini audit: skipped for live (rate-limited, every 10 frames)

---

## Planned Fix Summary

The fix targets 6 layers of the pipeline to reduce under-detection:
1. Lower S4 WBF worker IoU (prevent adjacent-worker merging)
2. Lower S4 post_gate thresholds (allow weaker fused detections through)
3. Lower recovery min_score + widen survival conditions
4. Lower is_valid_worker height/size/single-model thresholds for S4
5. Lower ValidWorkerValidator min_height_px for S4
6. Add condition-aware final threshold gate (S4 worker: 0.40 → 0.30)
