# Backup Notes — False Positive Fix
**Timestamp:** 2026-04-11  
**Purpose:** Pre-modification snapshot before false-positive suppression improvements

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

## Current Ensemble Thresholds (Pre-Fix)

### adaptive_postprocess.py — CONF_THRESHOLDS
| Condition | worker | helmet | safety_vest |
|-----------|--------|--------|-------------|
| S1_normal | 0.42 | 0.32 | 0.50 |
| S2_dusty | 0.32 | 0.18 | 0.30 |
| S3_low_light | 0.32 | 0.18 | 0.30 |
| S4_crowded | 0.38 | 0.22 | 0.34 |

### adaptive_postprocess.py — WORKER_MIN_HUMAN_SCORE
| S1_normal | S2_dusty | S3_low_light | S4_crowded |
|-----------|----------|--------------|------------|
| 0.43 | 0.38 | 0.34 | 0.30 |

### adaptive_postprocess.py — WORKER_MIN_PIXEL_HEIGHT
| S1_normal | S2_dusty | S3_low_light | S4_crowded |
|-----------|----------|--------------|------------|
| 24 | 22 | 20 | 18 |

### adaptive_postprocess.py — Worker aspect ratio
- Current: `0.15 <= ratio <= 1.25`  (too loose — allows wide objects)

### site_aware_ensemble.py — WBF post_gate (per condition)
| Condition | helmet | vest | worker |
|-----------|--------|------|--------|
| S1_normal | 0.22 | 0.24 | 0.24 |
| S2_dusty | 0.18 | 0.20 | 0.18 |
| S3_low_light | 0.16 | 0.18 | 0.16 |
| S4_crowded | 0.14 | 0.17 | 0.14 |

### site_aware_ensemble.py — recover_crowded_workers
- min_score: 0.22
- height filter: < 18 px (too permissive)
- aspect filter: > 1.15 (too permissive)
- score gate: >= 0.42

### site_aware_ensemble.py — TemporalPPEFilter
- Static filter: hits >= 3 AND human_score_ema < 0.36 AND no PPE streaks
- Max misses before track deletion: 6
- No motion/position variance tracking

---

## Current Active Model Weights
- YOLOv11: `weights/yolov11_buildsight_best.pt` (local fallback)
- YOLOv26: `weights/yolov26_buildsight_best.pt` (local fallback)
- SASTRA primary: `/nfsshare/joseva/.../best.pt`
- Model weights in WBF: YOLOv11=0.55, YOLOv26=0.45

---

## Gemini Auditor Status (Pre-Fix)
- `USE_GEMINI_AUDITOR = True` in pipeline_config.py
- Model: `gemini-2.5-flash`
- Current role: VALIDATE + SUPPLEMENT (adds new detections)
- Called on ALL detections (not just ambiguous ones)
- Supplement mode: can CREATE new worker/PPE boxes from Gemini output
- Daily free-tier quota: 20 requests/day (auto-disabled when exhausted)

---

## Known Issues Before Modification
1. **Cement bags detected as workers** — wide aspect ratio (0.8-1.2) passes the 1.25 limit; 
   human_score sometimes reaches 0.30-0.38 in S4 crowded mode
2. **Blue buckets detected as workers or safety vests** — color confusable with some vest colors; 
   squat boxy shape passes loose aspect ratio filter
3. **Scaffolding detected as workers** — vertical poles pass portrait aspect ratio check; 
   thin structures confuse edge-based human_score
4. **Material stacks creating duplicate worker detections** — no hard negative class suppression
5. **Static objects repeatedly triggering** — temporal filter threshold (human_score_ema < 0.36) 
   too loose; no position variance tracking
6. **Floating PPE detections** — PPE-to-worker overlap uses min_iou=0.05 which is extremely 
   loose; isolated vest/helmet on clutter passes through
7. **S4_crowded post_gate too low (0.14)** — crowded mode bleeds false positives into 
   normal scene frames that mis-classify as S4
8. **recover_crowded_workers too aggressive** — min_score=0.22 recovers far too many 
   non-worker detections
9. **Gemini in SUPPLEMENT mode** — currently adds new boxes, which can introduce phantom 
   worker/PPE detections; should be validator-only
