# Backup Notes - Pre-Rollback "Broken" State
**Date**: 2026-04-12
**Reason**: This backup captures the unstable state encountered during performance tuning and threshold hardening.

---

## 🛑 Primary Failure: "FAILED TO FETCH"
- **Behavior**: The dashboard intermittently shows "FAILED TO FETCH" on port 8000.
- **Root Cause**: Backend process (FastAPI) hanging on inference synchronization or a connection leak in the video processing loop. Port 8000 remains occupied but does not respond to health checks.

## 🔍 Detection & Logic Issues
- **Incorrect Dusty-Scene (S2) Activation**: The scene classifier was triggering S2 DUSTY on clear footage, likely due to overly aggressive brightness/contrast variance checks.
- **PPE Association Breakdown**: Many boxes are appearing as "Worker (Red)" even when PPE is clearly visible, or PPE boxes are appearing as "floating" orphans (separated from workers).
- **Cement Bag False Positives**: Static yellow/blue construction materials are still being flagged as safety vests/workers with confidence scores between 35-50%.

## ⚡ Performance & Artifacts
- **High Latency**: Significant lag between the video feed and the overlay boxes, making the system feel non-real-time.
- **Tile Artifacts**: Residual flickering or "grid" patterns visible during S4 crowded-scene processing.
- **Overlay Rendering**: Unstable drawing in `DetectionPanel.tsx` causing overlapping boxes and flickering bounding box colors.

---

## Recovery Target
Rolling back to: `dashboard/backend/backups/2026-04-11_realtime_zone_threshold_ppe_fix/`
This target represents the most stable ensemble state from April 11th.
