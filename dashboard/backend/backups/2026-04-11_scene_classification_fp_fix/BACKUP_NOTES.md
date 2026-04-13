# Backup Notes - 2026-04-11_scene_classification_fp_fix

Snapshot taken on 2026-04-11 12:00 PM.

## Included Files
- `server.py.bak`: Core backend/inference logic.
- `eval_gemini_auditor_slice.py.bak`: Evaluation script for Gemini auditor.
- `site_aware_ensemble.py.bak`: Video processing ensemble script.
- `DetectionPanel.tsx.bak`: Frontend Dashboard detections view.
- `DetectionPanel.css.bak`: Styling for the detection panel.

## Context
This backup was created before implementing the following optimizations:
1. `classify_scene_fast` enhancement (adding trigger reasons).
2. `SceneConditionTracker` hysteresis and transition logging.
3. `StaticClutterMask` and `suppress_floating_ppe` filters.
4. Gaussian Soft-NMS migration.
5. Atomic 5-tuple signature change for `run_inference`.
