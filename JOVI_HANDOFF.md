# Jovi — Handoff from Toni
**Date:** 2026-04-02
**From:** Toni
**Task:** Write the full BuildSight Project Report

---

## YOUR MISSION — Write the Full BuildSight Project Report

Write a comprehensive, professional technical report covering everything from the initial Indian dataset collection through Phase 2 completion and the post-processing pipeline.

**Output file:** `docs/buildsight_project_report.md`

---

## REPORT STRUCTURE — Follow This Exactly

### Title Block
```
# BuildSight AI — Full Project Report
**Project:** BuildSight — AI-Powered Construction Site PPE Detection
**Organisation:** Green Build AI
**Report Date:** 2026-04-02
**Prepared by:** Jovi (Gemini/Antigravity Agent) with Toni (Claude Sonnet 4.6)
**Status:** Phase 1–2.5 Complete | Phase 3 Ensemble Ready
```

---

### Section 1 — Project Overview
- What is BuildSight and why it exists (IGBC AP / NBC 2016 compliance context)
- The safety problem: PPE non-compliance on Indian construction sites
- System goal: real-time CCTV-based PPE detection
- 5-phase roadmap: Tournament → Ensemble → GeoAI (QGIS) → Live GIS Dashboard → SAMURAI tracking

---

### Section 2 — Indian Dataset Collection
- Source: real Indian construction site images collected by Green Build AI
- Total: 5,376 images across 4 site conditions
- **S1_normal** (1,373 images): baseline clear visibility
- **S2_dusty**: particulate matter, reduced contrast
- **S3_low_light**: early morning, indoor, evening scenes
- **S4_crowded**: dense worker clusters, heavy occlusion
- Synthetic data augmentation: AI-generated images to fill dataset gaps (named `synthetic_imagen_...`, `syn_dusty_...` etc.)
- Challenges: no initial annotations, class inconsistencies, varied camera hardware

---

### Section 3 — Annotation Pipeline
Key technical details to include:

- **Tool stack:** GroundingDINO (text-grounded detection) + SAM (Segment Anything Model)
- **3-class schema:** helmet (0), safety_vest (1), worker (2)
- **2-pass detection approach:**
  - Pass 1: Full-image worker detection using human-centric prompts to avoid machinery hallucinations
  - Pass 2: PPE sub-detection on each worker crop
- **Condition-specific preprocessing:** CLAHE + gamma for low-light, lower thresholds for dusty
- **Output formats:** YOLO .txt + COCO instances JSON + segmentation masks
- **Final annotation counts:** ~30,410 worker boxes, ~5,422 helmets, ~4,059 safety vests (training split)
- **Bugs fixed:** 5 annotation bugs addressed (two-pass detection, per-class NMS, per-class conf thresholds, low-light preprocessing, per-worker PPE sub-detection for crowds)

---

### Section 4 — Model Training
Cover all 4 models:

**YOLOv11:** Anchor-free, C2f backbone, 100 epochs on A100. Final mAP50=0.7381.
**YOLOv26:** Deeper anchor-free, 100 epochs. mAP50=0.7239. Stronger per-condition calibration.
**YOLACT++:** Real-time instance segmentation, 80,000 iterations. Final mAP50=0.5716. Technical bugs encountered during setup (Cython NMS NumPy deprecation, PyTorch constructor API, Config attribute missing, postprocess return value count) — all fixed before evaluation.
**SAMURAI:** Zero-shot video tracker. No PPE training. Deployed as ground truth reference and future tracking module (Phase 5).

---

### Section 5 — Phase 2: Tournament Evaluation (4×4 Grid)
- Protocol: 4 models × 4 conditions = 16 evaluation jobs on SASTRA A100
- Metrics: mAP50, precision, recall, F1, FPS, per-class AP
- Script: `val_condition_eval.py`

Copy the key results table from `docs/comparative_study.md` Sections 5.1 and 5.7. Include the full condition matrix. Key findings:
- YOLOv11 overall winner (mAP50=0.7381, F1=0.6725)
- YOLOv26 better per-condition calibration
- S2_dusty standout: YOLOv11 mAP50=0.8608 (best single-condition result)
- Safety vest hardest class: ~38% recall across all models
- YOLACT++ suitable only as supplementary segmentation tool
- SAMURAI confirmed as tracker-only, not standalone detector

---

### Section 6 — Phase 2.5: Post-Processing Pipeline

**Problem discovered after visual review of 2,764 annotated validation images:**
- Excavators/cranes detected as `worker` at confidence 0.63–0.88
- Safety vests detected on excavator yellow cabs
- Triple-stacked boxes per person (worker + vest + helmet separately)

**Root cause:** HSV color overlap between CAT yellow machinery and high-vis orange PPE; flat confidence threshold (0.25); no cross-class NMS.

**Adaptive post-processing system (v2) — 8 rules:**
Summarize from `docs/comparative_study.md` Section 14. Include the per-condition threshold table and rule effectiveness table.

**Results:**
- 48,025 → 23,352 detections (51.4% reduction)
- S1_normal: 64.2% FP reduction from machinery
- Helmet detection recovered via center-point containment anchor rule
- Residual ~15% excavator FPs — require retraining with machinery class

---

### Section 7 — Current System Status (as of 2026-04-02)

| Component | Status |
|-----------|--------|
| Indian Dataset | 5,376 images, annotated, split 80/10/10 |
| YOLOv11 | Trained, evaluated, post-processed |
| YOLOv26 | Trained, evaluated, post-processed |
| YOLACT++ | Trained, evaluated (supplementary role) |
| SAMURAI | Configured (tracking role, Phase 5) |
| Adaptive Post-Processing | v2 deployed on SASTRA, validated locally |
| Annotated Val Images | 1,382 v2 images available locally for review |
| Phase 3 Ensemble | Strategy defined in `docs/ensemble_strategy.md` |

---

### Section 8 — Phase 3 Preview: Multi-Model Ensemble Strategy
Summarize `docs/ensemble_strategy.md`:
- WBF fusion of YOLOv11 (weight=0.55) + YOLOv26 (weight=0.45)
- Per-class WBF IoU: worker=0.55, vest=0.50, helmet=0.45
- Post-WBF adaptive thresholds per condition
- Expected improvements: +3–5% mAP50, ~53% FN reduction in S2/S4, safety_vest recall 38%→48–52%
- Success criteria + early-exit FPS optimisation

---

### Section 9 — Known Limitations and Future Work

| Limitation | Impact | Proposed Fix |
|------------|--------|-------------|
| Machinery FPs (~15% residual) | False alarms in excavation sites | Retrain with `machinery` class (~300–500 annotations) |
| Safety vest recall ~38% | Missed compliance violations | Dataset expansion + class-specific augmentation |
| No mAP50-95 in condition eval | Incomplete COCO metric | Add pycocotools to val_condition_eval.py |
| SAMURAI needs video streams | Cannot evaluate on static val images | Phase 5 CCTV integration |
| No night/IR dataset | Limited low-light performance | Collect thermal/IR images |

---

### Section 10 — Appendix
- SASTRA infrastructure: node1, A100-PCIE-40GB, conda env at `/nfsshare/joseva/.conda/envs/buildsight/`
- ntfy notification topics: `buildsight-tournament-2026` (status), `buildsight-control-2026` (commands)
- Key scripts: annotate_indian_dataset.py, val_condition_eval.py, generate_condition_matrix.py, adaptive_postprocess.py, ensemble_inference.py, remote_notifier_v2.py, remote_listener_v2.py
- Key docs: comparative_study.md, ensemble_strategy.md, condition_eval_matrix.json

---

## SOURCE FILES — Read These Before Writing

```
docs/comparative_study.md          — Full Phase 2 results, all numbers
docs/ensemble_strategy.md          — Phase 3 plan
docs/condition_eval_matrix.json    — Per-condition metrics matrix
scripts/pipeline_config.py         — Annotation class schema and thresholds
TASK_STATE.json                     — Current project state
```

---

## WRITING STANDARDS

- Professional technical report style — proper paragraphs, not bullet dumps
- All numbers must match exactly what is in `comparative_study.md` — do not invent metrics
- Target length: 4,000–6,000 words
- Tables where appropriate, code blocks for config values
- You have full context on Phase 5 (SAMURAI tracking) and the GeoAI roadmap — include your perspective on those sections

---

## COMPLETION

When done, send ntfy:
```bash
curl -s -X POST \
  -H "Title: JOVI: Project Report Done" \
  -H "Priority: default" \
  -d "docs/buildsight_project_report.md complete. All 10 sections. Toni please review." \
  ntfy.sh/buildsight-tournament-2026
```

**Toni is monitoring.**
