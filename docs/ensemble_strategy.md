# BuildSight Phase 3 — Multi-Model Ensemble Strategy
**Authored by:** Toni (Claude Sonnet 4.6)
**Date:** 2026-04-02
**Status:** Ready for implementation

---

## 1. Objective

Combine YOLOv11 and YOLOv26 predictions using Weighted Box Fusion (WBF) to produce a single unified detection output that outperforms either model alone on all 4 site conditions.

**Target improvements over YOLOv11 solo baseline (mAP50=0.7381):**
- +3–5% mAP50 overall
- ~12% reduction in false negatives in S2_dusty and S4_crowded
- ~8% improvement in safety_vest recall (currently the weakest class at ~38%)

---

## 2. Why Weighted Box Fusion (WBF), Not NMS

Standard ensemble NMS (Non-Maximum Suppression) discards all but the highest-confidence box when overlap is detected. For PPE detection this is wrong — a helmet detected at 0.62 by YOLOv11 and 0.58 by YOLOv26 in the same location is strong evidence of a real helmet; NMS would keep only the 0.62 box and lose the corroborating evidence.

WBF instead **merges** overlapping boxes from all models into a single weighted-average box, incorporating confidence from all predictions. This gives:
- More accurate box coordinates (averaged, not winner-take-all)
- Higher fused confidence scores (combined evidence)
- Better recall for small/occluded items (safety vests, helmets in dusty conditions)

---

## 3. Ensemble Architecture

```
Input Image
    │
    ├── YOLOv11 (weight=0.55) → raw predictions: [box, cls, conf] × N
    │
    └── YOLOv26 (weight=0.45) → raw predictions: [box, cls, conf] × M
                │
                ▼
        WBF Fusion (per class)
        ─────────────────────
        IoU threshold:   0.55
        Skip threshold:  0.0001
        Weights: [0.55, 0.45]
                │
                ▼
        Post-WBF confidence gates (per class)
        ─────────────────────────────────────
        helmet:      ≥ 0.32
        safety_vest: ≥ 0.38
        worker:      ≥ 0.28
                │
                ▼
        Adaptive post-processing (8-rule system)
        ─────────────────────────────────────────
        Per-condition thresholds (Section 14, comparative_study.md)
        Vertical position filter (S1_normal excavation scenes)
        PPE worker-anchor containment check
                │
                ▼
        Final detections: {worker, helmet, safety_vest}
```

---

## 4. Model Weights Rationale

| Model | WBF Weight | Reason |
|-------|-----------|--------|
| YOLOv11 | **0.55** | Higher recall (0.6725 F1), stronger in S2_dusty (mAP50=0.8608), better on small workers in crowded scenes |
| YOLOv26 | **0.45** | Higher precision, better per-condition calibration (mean mAP50=0.3502 vs 0.3332 in per-condition eval), corrects YOLOv11 false positives |

The 55/45 split slightly favours YOLOv11 because recall is safety-critical — a missed worker without a helmet is a safety violation that must be flagged.

---

## 5. Per-Class WBF IoU Thresholds

Different classes have different spatial overlap characteristics:

| Class | WBF IoU | Reason |
|-------|---------|--------|
| worker | 0.55 | Workers have clear body boundaries; high IoU needed to merge genuine co-detections |
| safety_vest | 0.50 | Vest boxes vary with pose (bent-over workers have different vest visibility) |
| helmet | 0.45 | Helmet boxes are small and position varies; slightly looser merge threshold |

Implementation: run WBF separately per class, then combine results.

---

## 6. Pre-Fusion Confidence Gate

Before feeding predictions into WBF, apply a loose pre-filter at `conf ≥ 0.07` (same as adaptive postprocess inference gate). This ensures:
- Low-confidence corroborating detections from both models contribute to WBF fusion
- Very low noise detections (conf < 0.07) are excluded to prevent WBF from creating ghost boxes

---

## 7. Per-Condition Ensemble Configuration

The post-WBF adaptive thresholds should match the single-model adaptive system:

```python
POST_WBF_CONF = {
    "S1_normal":   {"worker": 0.55, "helmet": 0.32, "safety_vest": 0.50},
    "S2_dusty":    {"worker": 0.40, "helmet": 0.18, "safety_vest": 0.30},
    "S3_low_light":{"worker": 0.40, "helmet": 0.18, "safety_vest": 0.30},
    "S4_crowded":  {"worker": 0.45, "helmet": 0.25, "safety_vest": 0.40},
}
```

---

## 8. Expected Performance Gains

Based on error analysis from Phase 2:

### 8.1 False Negative Reduction
YOLOv11 misses workers that YOLOv26 catches and vice versa. WBF fusion captures both sets:

| Condition | YOLOv11 FN | YOLOv26 FN | Shared FN (both miss) | Expected Ensemble FN |
|-----------|-----------|-----------|----------------------|---------------------|
| S1_normal | 47 | 61 | ~22 | ~22 (−53% vs YOLOv11) |
| S2_dusty | 89 | 94 | ~42 | ~42 (−53%) |
| S3_low_light | 73 | 81 | ~35 | ~35 (−52%) |
| S4_crowded | 312 | 287 | ~148 | ~148 (−53%) |

### 8.2 Safety Vest Recall Boost
Safety vest is the hardest class (38% recall on YOLOv11). YOLOv26 has stronger vest precision. WBF merging of corroborating vest detections should push recall to ~48–52%.

### 8.3 Helmet Recall
Helmets benefit most from fusion in S2_dusty and S3_low_light where both models detect partial evidence at low confidence. WBF combines this into a single higher-confidence merged box.

---

## 9. Implementation Plan

### Step 1 — Update ensemble_inference.py on SASTRA
The script already exists at `/nfsshare/joseva/scripts/ensemble_inference.py`. Verify it implements WBF with the exact parameters above. If not, patch it.

### Step 2 — Run ensemble on val set (all 4 conditions)
```bash
/nfsshare/joseva/.conda/envs/buildsight/bin/python3 \
    /nfsshare/joseva/scripts/ensemble_inference.py \
    --conditions S1_normal S2_dusty S3_low_light S4_crowded \
    --output /nfsshare/joseva/val_annotated_ensemble
```

### Step 3 — Evaluate ensemble metrics
Run `val_condition_eval.py` on ensemble outputs using the same GT and protocol as Phase 2. Compare mAP50, F1, precision, recall per condition against single-model baselines.

### Step 4 — Update condition_eval_matrix.json
Add ensemble row to the matrix. Update comparative_study.md Section 5.7 with ensemble results.

### Step 5 — Visual review
Download `val_annotated_ensemble/` to local laptop and visually confirm:
- Excavator scenes (S1_normal): machinery FPs suppressed
- Dusty scenes (S2_dusty): helmets visible
- Crowded scenes (S4_crowded): dense group workers detected

### Step 6 — Threshold tuning (if needed)
If ensemble shows recall regression vs YOLOv11 solo, loosen WBF IoU threshold to 0.50 for all classes and re-run.

---

## 10. Success Criteria

The ensemble is considered successful if:

| Metric | Minimum threshold |
|--------|-------------------|
| Ensemble mAP50 overall | > 0.7381 (YOLOv11 solo) |
| Ensemble F1 | > 0.6725 (YOLOv11 solo) |
| safety_vest recall | > 0.42 (vs 0.38 solo) |
| S4_crowded F1 | > solo baseline |
| FPS (RTX 3090 equivalent) | > 18 FPS (real-time threshold) |

If ensemble FPS drops below 18 FPS due to dual-model inference, use **sequential inference with early-exit**: if YOLOv11 confidence for all detected objects > 0.80, skip YOLOv26 inference entirely. Only invoke YOLOv26 when YOLOv11 returns low-confidence or ambiguous results.

---

## 11. Fallback: If Ensemble Underperforms

If WBF ensemble does not beat YOLOv11 solo on mAP50:
1. Try NMS-based ensemble as comparison
2. Try increasing YOLOv11 weight to 0.65 / YOLOv26 to 0.35
3. Try class-specific weights: give YOLOv26 higher weight (0.60) for `safety_vest` only, where it has demonstrated stronger precision
4. If still underperforming: deploy YOLOv11 solo + adaptive post-processing as production system (already validated in Phase 2.5)

---

*Strategy authored by Toni (Claude Sonnet 4.6) — 2026-04-02*
*Based on Phase 2 metrics, error analysis, and per-condition deep analysis from comparative_study.md*
