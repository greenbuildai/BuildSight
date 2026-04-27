#!/bin/bash
# ============================================================
#  BuildSight Model Tournament — Sequential Training Script
#  SASTRA node1  |  GPU 1 = A100-PCIE-40GB
# ============================================================
set -e   # stop on first error

# ── Stage Jump Support ────────────────────────────────────────
# Usage: ./train_tournament.sh [START_STAGE]
# START_STAGE: 1=YOLOv11, 2=YOLOv26, 3=YOLACT++, 4=YOLACT++ (alias)
START_STAGE=${1:-1}
# Stage 4 is an alias for Stage 3 (YOLACT++) — no separate stage 4 training
[ "$START_STAGE" -eq 4 ] && START_STAGE=3

LOG="$HOME/tournament_log.txt"
echo "" > "$LOG"       # clear old log

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "==========================================================="
log "  BuildSight Multi-Model Sequential Training"
log "  Starting from Stage: $START_STAGE"
log "==========================================================="

# ── Environment ──────────────────────────────────────────────
log "Activating conda environment: buildsight"
source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate buildsight 2>>"$LOG"
log "Python: $(python --version 2>&1)"
log "Ultralytics: $(python -c 'import ultralytics; print(ultralytics.__version__)' 2>&1)"

DATA_ROOT="/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)"

# ── 1. YOLOv11 ───────────────────────────────────────────────
if [ "$START_STAGE" -le 1 ]; then
log ""
log "============ [1/3] YOLOv11 Training START ============"
cd "$DATA_ROOT/YOLOv11"
python train.py --model yolo11n.pt --epochs 100 --batch 16 --device 1 2>&1 | tee -a "$LOG"
log "============ [1/3] YOLOv11 Training DONE  ============"
else
log "[SKIP] Stage 1 (YOLOv11) — resuming from stage $START_STAGE"
fi

# ── 2. YOLOv26 ───────────────────────────────────────────────
if [ "$START_STAGE" -le 2 ]; then
log ""
log "============ [2/3] YOLOv26 Training START ============"
cd "$DATA_ROOT/YOLOv26"
python train.py --model yolo26n.pt --epochs 100 --batch 16 --device 1 2>&1 | tee -a "$LOG"
log "============ [2/3] YOLOv26 Training DONE  ============"
else
log "[SKIP] Stage 2 (YOLOv26) — resuming from stage $START_STAGE"
fi

# ── 3. YOLACT++ ──────────────────────────────────────────────
# The bundled train.py only prints instructions; it does NOT
# actually train.  We must clone the official YOLACT repo and
# configure it ourselves.
log ""
log "============ [3/3] YOLACT++ Training START ============"
YOLACT_DIR="$HOME/yolact"

if [ ! -d "$YOLACT_DIR" ]; then
    log "Cloning YOLACT++ repository..."
    cd "$HOME"
    git clone https://github.com/dbolya/yolact.git 2>&1 | tee -a "$LOG"
    cd "$YOLACT_DIR"
    pip install -r requirements.txt 2>&1 | tee -a "$LOG"
else
    log "YOLACT++ repo already exists at $YOLACT_DIR"
fi

# Inject BuildSight dataset config into yolact/data/config.py
YOLACT_DATA="$DATA_ROOT/YOLACT_plusplus"
ANN_DIR="$YOLACT_DATA/annotations"
IMG_DIR="$YOLACT_DATA/images"

# Append config if not already present
if ! grep -q "BuildSight" "$YOLACT_DIR/data/config.py" 2>/dev/null; then
    log "Injecting BuildSight dataset config into yolact..."
    cat >> "$YOLACT_DIR/data/config.py" << PYEOF

# ----- BuildSight (auto-injected) -----
buildsight_dataset = dataset_base.copy({
    'name': 'BuildSight',
    'train_images': '${IMG_DIR}/train',
    'train_info':   '${ANN_DIR}/instances_train.json',
    'valid_images': '${IMG_DIR}/val',
    'valid_info':   '${ANN_DIR}/instances_val.json',
    'has_gt': True,
    'class_names': ('helmet', 'safety_vest', 'worker'),
    'label_map': {1: 1, 2: 2, 3: 3},
})

buildsight_config = yolact_plus_resnet50_config.copy({
    'name': 'buildsight',
    'dataset': buildsight_dataset,
    'num_classes': 4,   # 3 classes + 1 background
    'max_iter': 80000,
})
PYEOF
fi

cd "$YOLACT_DIR"
python train.py --config=buildsight_config --batch_size=8 2>&1 | tee -a "$LOG"
log "============ [3/3] YOLACT++ Training DONE  ============"

# ── 4. SAMURAI ───────────────────────────────────────────────
log ""
log "[4/4] SAMURAI is zero-shot — no training required."

log ""
log "==========================================================="
log "  ALL TRAINING COMPLETE! Ready for Evaluation."
log "==========================================================="
