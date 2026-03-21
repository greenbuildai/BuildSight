# BuildSight AI — Cross-AI Coordination Handshake
**Project**: BuildSight Indian Construction Site PPE Detection Dataset
**Claude Version**: claude-sonnet-4-6
**Antigravity**: Google Gemini (Antigravity)
**Last Updated**: 2026-03-20
**Status**: READY FOR RE-ANNOTATION RUN

---

## PURPOSE
This file is the shared state document between Claude (Anthropic) and Antigravity (Google).
The human operator carries this file between both AI sessions.
Each AI reads this file, acts on its assigned tasks, then writes its output back here.

---

## 1. LOCKED SCHEMA — DO NOT CHANGE

```
Class ID 0 → helmet       (includes hard_hat, hard hat)
Class ID 1 → safety_vest  (includes high_vis_jacket, high vis)
Class ID 2 → worker       (full body bounding box)

NUM_CLASSES = 3
safety_boots = PERMANENTLY REMOVED (poor detection quality, user decision 2026-03-20)
```

**This schema is locked. Any AI that changes this will corrupt the dataset.**

---

## 2. CURRENT STATE (written by Claude)

### What has been completed:
- [x] 3-class schema enforced across ALL pipeline scripts
- [x] `pipeline_config.py` — 3-class, correct thresholds
- [x] `annotate_indian_dataset.py` — 3-class, COCO_CATEGORIES fixed
- [x] `export_for_models.py` — 3-class, both data.yaml templates fixed
- [x] `validate_annotations.py` — safety_boots fully removed
- [x] `visualize_annotations.py` — color map updated to 3-class
- [x] 3 detection improvement fixes implemented (see Section 4)
- [x] Crowded_Condition partial run: 687/991 images annotated (terminated by user)
- [x] All existing Normal/Dusty/LowLight label files present (from prior run)

### What needs to be done:
- [ ] Full re-annotation of all 4 conditions with improved pipeline
- [ ] Run `merge_crowded_annotations.py` after pipeline completes
- [ ] Run `export_for_models.py` to refresh all 4 model export formats
- [ ] Run `validate_annotations.py` to confirm improved H/W% and V/W% rates

---

## 3. PIPELINE COMMAND TO RUN

```bash
cd "e:\Company\Green Build AI\Prototypes\BuildSight"
.venv\Scripts\python.exe scripts/annotate_indian_dataset.py --skip-sam --no-auto-detect 2>&1
```

**Flags:**
- `--skip-sam` → skip SAM segmentation mask generation (speed, ~10s saved/img)
- `--no-auto-detect` → trust folder condition directly, skip DINO condition-detection pass (~3s saved/img)
- No `--conditions` flag → runs ALL 4 conditions (Normal, Dusty, LowLight, Crowded)

**Hardware**: RTX 4050 6GB VRAM, CUDA enabled
**Estimated time**: ~6–8 hours for all 4 conditions
**Expected throughput**: ~8–12 images/minute

---

## 4. DETECTION IMPROVEMENTS IMPLEMENTED (by Claude)

### Fix 1 — Decoupled DINO Thresholds
| Pass | Threshold | Reason |
|------|-----------|--------|
| Pass 1: Full image, worker detection | `DINO_BOX_THRESHOLD = 0.28` | Strict — only confident workers |
| Pass 2: Crop-level PPE detection | `DINO_CROP_BOX_THRESHOLD = 0.10` | Aggressive — tiny crops can afford it |

### Fix 2 — 2x Crop Upscaling (Magnifying Glass)
```python
WORKER_CROP_UPSCALE_FACTOR = 2.0
```
Every worker crop is doubled in resolution before being fed to GroundingDINO.
Reason: ViT patches are 16×16px. A 60px-tall worker crop only gives 4 patches for the helmet.
Doubling to 120px gives 8 patches — detection confidence measurably improves.

### Fix 3 — Simplified PPE Prompt
```python
PPE_CROP_TEXT_PROMPT = "helmet . vest ."
```
Short, direct, no ambiguous terms ("hat" or "jacket" were rejected — too broad, cause false positives).

### Fix B2 — Full-Image PPE Pass for Crowded Scenes
For images with ≥5 workers detected, an additional full-image DINO pass runs PPE detection
at full 1080p resolution, with results filtered by worker proximity (association check).
This compensates for tiny worker crops in densely packed scenes.

---

## 5. KEY CONFIG VALUES (from pipeline_config.py)

```python
# Thresholds
CONFIDENCE_THRESHOLDS = {"helmet": 0.20, "safety_vest": 0.20, "worker": 0.40}
CONDITION_THRESHOLD_DELTA = {"normal": 0.00, "crowded": 0.00, "dusty": -0.05, "low_light": -0.07}
DINO_BOX_THRESHOLD       = 0.28   # Pass 1 full image
DINO_TEXT_THRESHOLD      = 0.15
DINO_CROP_BOX_THRESHOLD  = 0.10   # Pass 2 crop
DINO_CROP_TEXT_THRESHOLD = 0.10

# Prompts
WORKER_TEXT_PROMPT    = "person . worker . construction worker . man ."
PPE_TEXT_PROMPT       = "helmet . vest ."
PPE_CROP_TEXT_PROMPT  = "helmet . vest ."

# Crop settings
WORKER_CROP_PADDING        = 0.20
WORKER_CROP_UPSCALE_FACTOR = 2.0
WORKER_MIN_HEIGHT_FOR_PPE  = 40

# NMS
NMS_IOU_THRESHOLD = {"helmet": 0.45, "safety_vest": 0.45, "worker": 0.50}

# Crowded B2 trigger
CROWDED_SCENE_THRESHOLD = 5
```

---

## 6. OUTPUT STRUCTURE

```
Dataset/Final_Annotated_Dataset/
├── images/
│   ├── train/   (70%)
│   ├── val/     (20%)
│   └── test/    (10%)
├── labels/          ← YOLO .txt format (class cx cy w h per line)
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/     ← COCO JSON format (for SAMURAI + YOLACT++)
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
└── data.yaml        ← nc=3, names: [helmet, safety_vest, worker]
```

---

## 7. MODEL COMPATIBILITY

| Model | Format Required | Status |
|-------|----------------|--------|
| YOLOv11 | YOLO .txt bbox | Compatible |
| YOLOv26 | YOLO .txt bbox (±3px tight) | Compatible |
| SAMURAI | COCO JSON + track_id + sequence_id | Compatible |
| YOLACT++ | COCO JSON + segmentation polygons | SAM skipped — polygons need post-processing |

**Note for Antigravity**: SAM was skipped for speed (`--skip-sam`). YOLACT++ requires polygon masks.
After re-annotation, a SAM-only post-pass is needed for Crowded images if polygon masks are required.

---

## 8. TASK DIVISION (PROPOSED)

| Task | Owner | Status |
|------|-------|--------|
| Pipeline code + config fixes | Claude | DONE |
| Launch & monitor full re-annotation run | Either AI or user | PENDING |
| Merge crowded annotations post-run | Claude | PENDING |
| Export for all 4 models | Claude | PENDING |
| Quality audit (H/W%, V/W% rates) | Claude | PENDING |
| SAM post-pass for YOLACT++ | TBD | PENDING |

---

## 9. ANTIGRAVITY — WRITE YOUR STATUS HERE

*(Antigravity: update this section with what you have done, what you are doing, and any issues found)*

```
Antigravity status: [AWAITING INPUT]
Last action:
Current task:
Issues found:
```

---

## 10. PREVIOUS AUDIT RESULTS (baseline to beat)

| Condition | H/W% | V/W% | Notes |
|-----------|------|------|-------|
| Normal | TBD | TBD | Prior run — no B2 |
| Dusty | TBD | TBD | Prior run — no B2 |
| Low Light | TBD | TBD | Prior run — no B2 |
| Crowded | 18% | 16% | Prior run — B2 now implemented |

Target after re-annotation with 3 fixes: H/W% > 50%, V/W% > 45% across all conditions.
