# Handoff to Claude (Toni)

**Task ID**: HANDOFF-004
**Status**: ready_for_claude
**Assigned To**: Toni (Claude)

## Objective
Eliminate false positive helmet detections on bare heads, wrists, and hands by implementing a refined multi-gate validation pipeline.

## Context
Recent recall optimizations (2026-04-11) for crowded scenes (`S4_crowded`) have introduced significant precision regressions. The model is frequently misclassifying skin-toned round regions (bare heads, wrists, palms) as hard hats.

A backup of the current stable/baseline logic has been created at `backups/backup_20260412_0806/`.

## Implementation Plan (Refined)

### 1. Helmet Validation Layer (`scripts/adaptive_postprocess.py`)
Implement a `HelmetValidationLayer` with the following gates:
- **`_gate_vertical_ratio`**: Helmets MUST be located within the top **35%** of the associated worker's bounding box height.
- **`_gate_skin_tone` (Penalty-Based)**: 
  - **IMPORTANT**: Use skin-tone rejection as a **penalty**, not a hard reject.
  - **Preserve**: Orange, Yellow, Faded White, Beige, and Dusty helmets.
  - **Hard Reject ONLY if**: 
    - Skin-tone confidence is high (HSV range overlap with flesh tones).
    - Texture looks soft/skin-like (low Laplacian/Sobel edge density).
    - Helmet is outside the upper head zone (>35% from top).
    - Worker association score is weak.
- **`_gate_texture`**: Use edge density to separate smooth hard hats from hair or cloth textures.

### 2. Strengthened Association Logic
- Update `has_worker_overlap` in `scripts/adaptive_postprocess.py` to enforce the vertical constraint.
- Tighten worker-to-helmet association rules to prevent "floating" helmet detections.

### 3. Ensemble Integration (`scripts/site_aware_ensemble.py`)
- Apply the same validation rules consistently across the WBF ensemble.
- Refine `_worker_has_ppe` and `_copy_ppe_from_track` to ensure synthetic helmet placement adheres to the new geometric and skin-tone constraints.

## Verification Plan
1. Run the `run()` validation test in `adaptive_postprocess.py` to compare precision/recall against the baseline.
2. Verify that **Orange, Yellow, and White** helmets are NOT suppressed.
3. Confirm that wrist/palm detections are eliminated.

## Required Output (in GEMINI_HANDBACK.md)
- **Status**: success / failure
- **Summary**: Precision improvement % and impact on recall for crowded scenes.
- **Files Modified**: `scripts/adaptive_postprocess.py`, `scripts/site_aware_ensemble.py`.
- **Open Issues**: Any remaining corner cases (e.g., extremely low-light skin vs brown helmet confusion).
