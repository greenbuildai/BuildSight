# BuildSight Ensemble Backend Backup (Pre-Refactor)

Timestamp: 2026-04-10
File backed up: `dashboard/backend/server.py`
Reason: preserve pre-refactor ensemble backend behavior before unifying runtime logic with shared scripts.

## Pre-refactor runtime characteristics

- Local backend-owned logic for:
  - Weighted box fusion (`wbf_fuse`)
  - Worker false-positive filtering (`is_valid_worker`)
  - Video temporal filtering (`WorkerMemory`, `ServerSideTracker`)
- Endpoint paths using this local logic:
  - `POST /api/detect/image`
  - `POST /api/detect/frame`
  - `POST /api/detect/video`

## Pre-refactor key parameters snapshot

- `MODEL_WEIGHTS = [0.55, 0.45]`
- `WBF_IOU = {0: 0.45, 1: 0.50, 2: 0.65}`
- `POST_WBF_GLOBAL = {0: 0.30, 1: 0.14, 2: 0.18}`
- `PRE_CONF = 0.20`
- `NMS_IOU = 0.60`
- `EARLY_EXIT_CONF = 0.75`
- `POST_WBF_BY_CONDITION`:
  - `S1_normal: {0: 0.30, 1: 0.14, 2: 0.18}`
  - `S2_dusty: {0: 0.25, 1: 0.10, 2: 0.15}`
  - `S3_low_light: {0: 0.22, 1: 0.10, 2: 0.14}`
  - `S4_crowded: {0: 0.28, 1: 0.12, 2: 0.15}`

## Rollback note

If rollback is needed, restore `dashboard/backend/server.py` from git history prior to this refactor.
