# Backup Notes — Real-Time, Zone Filtering, PPE Accuracy, Clutter Suppression Fix
**Date**: 2026-04-11  
**Purpose**: Snapshot before optimizing real-time performance, zone filtering, PPE validation, and material suppression

---

## Current FPS and Latency (Pre-Fix)

- Inference gap: MIN_GAP_MS = 80ms between loop iterations
- GPU inference time: ~100–150ms per frame (RTX 4050, FP16)
- Total round-trip: ~150–280ms
- S4 tile inference adds ~100–120ms (4 tiles)
- Gemini audit: rate-limited to every 10 frames
- LiveMode: setState on EVERY frame → React re-renders ~10x/second

---

## Current Crowded-Scene Logic (Post recall-fix)

- S4 profile: pre_conf=0.07, wbf_iou[WORKER]=0.28, post_gate[WORKER]=0.14
- Recovery: min_score=0.20, elevated-zone path added, PPE search ±25%
- Tile inference: ALWAYS runs for S4 (even when 4+ workers detected)
- Worker validator: min_height_px temporarily lowered to 14 for S4

---

## Current Worker / Helmet / Vest Thresholds

```python
THRESHOLD_STATE (defaults):
    global_floor: 0.35
    worker:       0.40
    helmet:       0.45
    vest:         0.35

S4 overrides in run_inference:
    worker: min(threshold, 0.28)
    helmet: min(threshold, 0.22)
    vest:   min(threshold, 0.22)
```

---

## Current Site-Zone Filtering Behavior

- Zone filtering is FRONTEND ONLY (canvas draw loop, RAF tick)
- Point-in-polygon (ray casting) on detection centroid in letterbox-normalised coords
- Backend receives NO zone polygon — processes all detections regardless of zone
- Out-of-zone detections are visible briefly before RAF loop removes them
- Zone polygon not sent in form data to detect_frame endpoint

---

## Current PPE Validation Logic

### Helmet validation (`is_valid_helmet`):
- Aspect ratio: 0.45–2.2
- Width ≤ 70% of worker box width
- Height ≤ 55% of worker box height
- Min size: 8×8 px
- Score gate: THRESHOLD_STATE["helmet"]

### Vest validation (`associate_ppe_to_workers`):
- Centroid must fall in vest zone: (bx1–pad, bx2+pad) × (by1, by2+20% below)
- No aspect ratio check
- No color check
- No isolation rejection (orphaned vests kept in output)

### Orphaned PPE:
- Helmets/vests with no nearby worker are NOT filtered — they remain in final output
- Floating helmets and vests appear when no worker is associated

---

## Current False-Positive Issues

1. Yellow cement bags detected as safety vests (color confusion)
2. Kerchiefs/cloth wraps detected as helmets (soft texture fools model)
3. Floating helmet/vest detections with no associated worker
4. Blue buckets and stacked cement bags occasionally passing worker filters
5. Static clutter sometimes persisting across frames before motion gate catches it

---

## Current Known Problems

- Vest: no isolation rejection → yellow bags that score ≥ vest threshold appear
- Helmet: aspect ratio cap 2.2 is loose → long cloth wraps pass
- Zone: backend processes all detections → wasted GPU cycles on out-of-zone areas
- LiveMode: `setDetections([...data.detections])` on every frame → React re-renders at 10Hz
- Zone polygon not sent to backend → no server-side filtering

---

## Reason for This Backup

This backup captures the improved S4 worker recall state (2026-04-11 morning changes) before applying:
1. Backend zone polygon filtering
2. Orphaned PPE removal (`filter_orphan_ppe`)
3. Stronger vest validation (aspect + isolation requirement)
4. Stronger helmet validation (tighter aspect ratio)
5. Stronger material color suppression for vest false positives
6. LiveMode React re-render throttling
7. Frontend zone polygon forwarding to backend
