# Gemini Auditor Integration Backup
Date: 2026-04-10

Backed up before wiring Gemini auditor into dashboard/backend/server.py.

Contents:
- server.py.bak
- site_aware_ensemble.py.bak
- ensemble_inference.py.bak
- est.pt.bak (active local fallback weight present in workspace)

Notes:
- Workspace does not currently contain weights/yolov11_buildsight_best.pt or weights/yolov26_buildsight_best.pt.
- The live backend in this workspace would currently fall back to the local est.pt if started as-is.
