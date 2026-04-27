# YOLO PPE Model Analysis Report: Construction Site Robustness & Spatial Risk

**Date:** 2026-02-18
**Project:** BuildSight Prototypes
**Status:** Performance Baseline Established

---

## 1. Executive Summary
This report analyzes the performance of the YOLO-based Personal Protective Equipment (PPE) detection model across various construction site conditions (S1-S4). While the model demonstrates high precision in baseline environments, it reveals significant generalization gaps in challenging operational scenarios. Addressing these limitations is critical before integrating detection outputs into GIS-based spatial risk maps.

---

## 2. Quantitative Performance Summary
The model was evaluated across two primary training phases. The 100-epoch refined model demonstrates a substantial improvement in recall and overall detection accuracy.

| Metric | 40-Epoch Baseline | 100-Epoch Refined (`S4-temp`) | Delta |
| :--- | :--- | :--- | :--- |
| **Precision (B)** | 0.7516 | **0.8999** | +0.1483 |
| **Recall (B)** | 0.6682 | **0.7311** | +0.0629 |
| **mAP50 (B)** | 0.6941 | **0.8088** | +0.1147 |
| **mAP50-95 (B)**| 0.4603 | 0.5071 | +0.0468 |

### **Observation:**
The model remains **Precision-heavy**. While it rarely misidentifies an object (90% Precision), it still misses nearly 27% of PPE instances (73% Recall), particularly in Crowded (S4) and Low-Light (S3) scenarios.

---

## 3. Qualitative Performance Issues & Root Causes

### **A. "Loose" Detections (Person-Equipment Dissociation)**
*   **Issue:** The model detects a "Helmet" or "Safety-vest" in the frame but fails to anchor it to a "Person" entity.
*   **Root Cause:** Bounding box overlap is too small, or the person is partially occluded, leading the model to see the PPE as an isolated object.
*   **Impact:** GIS risk maps might show "Vests" floating in space without workers, leading to skewed spatial density metrics.

### **B. "Loose" Persons (Unprotected Worker Misclassification)**
*   **Issue:** A worker is detected as a "Person" with high confidence, but their PPE is either missed or flagged as "Head" (No Helmet).
*   **Root Cause:** Motion blur in site footage or low resolution in wide-angle CCTV views.
*   **Priority:** **CRITICAL**. This is the primary failure mode for safety enforcement.

### **C. Bounding Box Instability (Jitter)**
*   **Issue:** Detections flicker on and off across consecutive frames.
*   **Root Cause:** Lack of temporal consistency in a frame-by-frame detector (YOLO alone).
*   **Suggested Fix:** Implement **ByteTrack** or **DeepSORT** for detection smoothing.

---

## 4. Site Condition Comparative Analysis (S1-S4)

| Scenario | Condition | Observed Performance | Risk Level |
| :--- | :--- | :--- | :--- |
| **S1** | Normal Baseline | Optimal. High confidence across all 11+ classes. | Low |
| **S2** | Dusty / Hazy | Confidence drops by ~15-20%. Small objects like "Gloves" are lost. | Medium |
| **S3** | Low-Light | Significant false negatives for "Safety-vests" due to lack of color contrast. | High |
| **S4** | Crowded | Overlapping bounding boxes lead to "Person" merging. | High |

---

## 5. Generalization Gaps
*   **Annotation Variance:** Current training data (SHWD/VOC) relies heavily on high-contrast frontal views. In real-world deployment, drone footage provides top-down views where typical "Person" features overlap with helmet boundaries.
*   **Operational Latency:** Inference speed is ~7-10ms (GPU), which is sufficient, but post-processing of Crowded (S4) scenes adds overhead that must be optimized for real-time GIS updates.

---

## 6. Recommendations for Spatial Risk Integration

To ensure the model is robust for **GIS-based spatial risk integration**, the following technical improvements are mandatory:

1.  **Hierarchical Verification (The "BuildSight Method"):**
    *   Do not trigger an alarm unless a `Person` detection has child `Helmet` and `Vest` detections within its bounding box ROI.
2.  **Contrast Enhancement (S3 Handling):**
    *   Integrate a pre-inference "Dehaze" or "Low-Light Enhancement" block for S2/S3 site feeds.
3.  **Spatial Buffering:**
    *   When projecting detections onto GIS maps, use a "Temporal Buffer" (e.g., a detection must persist for 5 frames) to avoid false alerts on risk maps.
4.  **Dataset Balancing:**
    *   Re-train with a focus on **Person-Hardhat-Vest** triplets rather than isolated helmet detection to improve the Person-PPE association.

---
**Report Compiled By:** Antigravity AI
**Next Steps:** Proceed to Phase 2 - Integration with GIS Spatial Mapping Service.
