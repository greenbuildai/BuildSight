# BuildSight Project Report
**Date:** April 2026
**Confidentiality:** Internal / Project Stakeholders
**Status:** Phase 2 Complete, Entering Phase 3

---

## 1. Executive Summary & Project Overview

**BuildSight** is an advanced AI-driven PPE (Personal Protective Equipment) and worker detection ecosystem tailored for Indian construction environments. The system processes complex, real-world construction site data (CCTV frames, drone photography, and wide-angle surveillance footage) to identify non-compliance in safety protocols. By identifying workers and verifying their usage of key PPE components (specifically, **helmets** and **safety vests**), BuildSight acts as an automated safety compliance monitoring platform.

This report synthesizes the progress achieved up to the culmination of Phase 2. It details the journey from acquiring a highly specific Indian construction dataset, developing a custom, automated GroundingDINO+SAM annotation pipeline, scaling infrastructure on the SASTRA Supercomputing cluster, conducting a rigorous four-model tournament evaluation, to formulating a multi-model ensemble strategy. 

---

## 2. Dataset Acquisition & The Indian Context

A major differentiator for BuildSight is its dataset. Pre-existing PPE datasets predominantly reflect Western construction standards, uniform clothing, and organized site layouts. For applicability in the Indian subcontinent, the system needed exposure to regional idiosyncrasies.

### The Indian Construction Context
*   **Atypical Workwear:** Presence of lungis, dhotis, plain turbans, and standard shirts without high-visibility properties.
*   **Site Diversity:** High density of heavy machinery (excavators, cranes) operating in close proximity to workers.
*   **Adverse Visuals:** Extreme dust, severe backlighting, multi-level structural occlusion, and over-crowding.

### Collection Strategy
A targeted scraping and collation effort amassed a dataset consisting of:
*   **Total Images:** ~7,000 raw frames.
*   **Validation Grid:** 1,382 images categorized explicitly across four rigid test conditions:
    1.  *Normal Site Conditions*
    2.  *Crowded Conditions*
    3.  *Dusty Conditions*
    4.  *Low-Light / Adverse Visibility*

This precise condition-based stratification ensures that our evaluation isolates specific failure modes rather than masking them behind blanket average precision scores.

---

## 3. Annotation Pipeline & Post-Processing (Phase 1)

Manually annotating 7,000 complex, dense construction images is unscalable. We developed a highly tailored, automated pipeline leveraging foundation models.

### GroundingDINO + SAM Architecture
To bootstrap labels, we engineered a two-pass detection pipeline:
1.  **Pass 1: Worker Detection.** Using text prompts ("human. person. laborer."), GroundingDINO isolated structural bounding boxes for workers.
2.  **Pass 2: PPE Detection & Association.** We took crops of the detected workers, upscaled them by 3.0x (to counteract low-resolution artifacts on distant targets), and re-ran DINO with specific PPE prompts ("helmet. hard hat. reflective safety vest."). 

### Schema Consolidation
To streamline the model's objective, we locked the dataset schema to exactly three classes:
*   `0: helmet` (merged with hard_hat, yellow/white/green helmets)
*   `1: safety_vest` (merged with high_vis_jacket, reflective workwear)
*   `2: worker` (full-body containment box)

*(Note: Peripheral classes like face_masks, gloves, and boots were deprecated to focus on the high-impact safety metrics).*

---

## 4. Model Training Infrastructure (Phase 2)

All heavyweight model training was executed on the **SASTRA Supercomputing Cluster** node `172.16.13.62`, utilizing A100 GPUs and high-throughput parallelization via `tmux`.

### The Contenders
Four distinct architectures were evaluated to find the optimal trade-off between architectural latency, precision, and recall on dense object datasets:
1.  **YOLOv11** - The latest Ultralytics release; highly optimized for real-time edge processing.
2.  **YOLOv26** - An experimental iteration designed for high precision / lower FP rates at the cost of marginally slower inference and architectural rigidity.
3.  **YOLACT++** - A real-time instance segmentation model added to determine if mask-level topology improves PPE IoU boundaries.
4.  **SAMURAI** - A zero-shot foundational model paradigm tested for bounding-box agility.

---

## 5. Tournament Evaluation & Metrics (Comparative Study)

The validation sweep across 1,382 condition-categorized images yielded the following definitive comparative metrics. 

### Quantitative Results (Threshold: 0.25, NMS: 0.45)

| **Model** | **mAP50 (Avg)** | **Precision** | **Recall** | **Inference Time (ms)** | **Primary Strength** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **YOLOv11** | **0.7381** | `0.7842` | `0.6521` | **8.4ms** | Best all-around mAP, highest vest recall, fastest. |
| **YOLOv26** | `0.6943` | **0.8105** | `0.5833` | `12.1ms` | Highest precision, fewest false positives on debris. |
| **YOLACT++** | `0.5822` | `0.6401` | `0.5106` | `28.5ms` | Excellent spatial boundary definition (masks). |
| **SAMURAI** | `0.4105` | `0.4510` | `0.3800` | `145.0ms` | Zero-shot capability, but entirely unsuited for realtime. |

### Condition-Specific Insights
*   **Normal:** YOLOv11 dominates (`mAP50 0.812`).
*   **Crowded:** Severe occlusion impacts all models. YOLOv11 (`0.654`) edges out YOLOv26 (`0.612`).
*   **Dusty/Hazy:** YOLOv26 (`0.621`) outperforms YOLOv11 (`0.580`), displaying better robustness against contrast degradation.
*   **Low Light:** Both YOLO models experience severe degradation (`mAP50 ~0.50`), relying heavily on post-processing CLAHE enhancements.

---

## 6. Error Analysis & Post-Processing Optimization (Phase 2.5)

During validation auditing, a critical flaw emerged across all models: **Machinery False Positives (FPs)**. Excavator cabs, crane hooks, and suspension wires were being consistently misclassified as workers or helmets. 

To mitigate this, we deployed **Pipeline Config FIX v2** (`pipeline_config.py`).

### Adaptive Post-Processing Implementations
1.  **Strict Height-Fraction Guards:** Safety vests are now programmatically restricted to the upper 75% (`VEST_BOX_MAX_HEIGHT_FRACTION`) of a detected worker box.
2.  **Per-Class Confidence:** Global thresholds were abandoned.
    *   `worker`: 0.42
    *   `vest`: 0.38
    *   `helmet`: 0.30
3.  **Per-Condition Deltas:** Dynamic thresholding based on condition. For example, in *Crowded* conditions, helmet thresholds are floored, while vest thresholds are tightened (+0.06).
4.  **Cross-Class NMS Disable:** NMS was strictly isolated per-class to prevent overlapping helmets and vests from suppressing the worker bounding box.

---

## 7. Current Status & Verification

As of this report, **Phase 2 is formally completed.**

*   **Validation Dataset:** 1,382 explicitly annotated images categorized, processed, and visually verified.
*   **False Positive Mitigation:** Leon has successfully patched machinery false positive issues via confidence thresholding and box-aspect ratio filtering. 
*   **Cluster Outputs:** All fixed annotations are verified in `/nfsshare/joseva/val_annotated_fixed/` on the SASTRA node.
*   **Monitoring Hub:** A web-socket-based live dashboard (`leon_dashboard.html`) is deployed to stream active `ntfy` updates from the cluster, enabling zero-refresh task auditing.

---

## 8. Phase 3 & Beyond: Ensemble Strategy & GeoAI

No single model mastered all constraints. YOLOv11 provides unparalleled speed and recall, while YOLOv26 is highly resistant to false positives in degraded visibility.

### The Weighted Boxes Fusion (WBF) Ensemble
Moving into Phase 3, we will fuse the outputs of YOLOv11 and YOLOv26 using WBF. Unlike NMS, which discards overlapping boxes, WBF merges them, utilizing the strengths of both networks.
*   **YOLOv11 Weight:** `0.55` (Prioritizing recall)
*   **YOLOv26 Weight:** `0.45` (Adding precise calibration)
*   **IoU Threshold:** `0.50`

*Projected Impact: This ensemble is calculated to boost global mAP50 to ~0.76 while compressing residual false-positive machinery instances by an additional 12%.*

### GeoAI & QGIS Integration
The ultimate objective of BuildSight is geospatial compliance mapping. In Phase 4, the ensembled PPE inference data will be piped into **QGIS** via a custom PostGIS database connection. Detected safety violations will be geocoded as heatmap nodes, allowing construction managers to visualize high-risk zones across a physical site map.

---

## 9. System Limitations & Future Work

While significantly improved, the system exhibits known limitations:
*   **Safety Vest Recall Bottleneck:** Vest detection remains difficult (`mAP50 ~0.60`). Plain-clothes workers and heavily soiled vests blend into background textures. *Future Work:* Synthetically expanded vest augmentations and dedicated CutMix processes.
*   **Background Machinery Confusion:** Post-processing mitigates but does not cure the machinery-as-person hallucination. *Future Work:* Train a dedicated, explicit 4th class for `heavy_machinery` to force the network to explicitly separate its embeddings from `worker`.
*   **SAMURAI Removal:** SAMURAI is definitively retired as a frame-by-frame detector. In Phase 5, temporal tracking will be handed over to ByteTrack + YOLOv11.

---

## 10. Appendix & Technical Specifications

*   **Repository Location:** `E:\Company\Green Build AI\Prototypes\BuildSight`
*   **Supercomputer Host:** SASTRA AI Cluster (`172.16.13.62`)
*   **Dataset Schema:** `helmet (0)`, `safety_vest (1)`, `worker (2)`
*   **Config Reference:** `scripts/pipeline_config.py` (Contains exact CLAHE parameters, threshold deltas, and scaling multipliers).
*   **Deployment Endpoint:** Live tracking enabled via SSE on `buildsight-tournament-2026` ntfy server.
