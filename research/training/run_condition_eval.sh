#!/bin/bash
# Wait for YOLACT++ training (PID 154709) to finish, then run per-condition eval
source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate
conda activate buildsight

echo "[condition_eval] Waiting for training process 154709 to finish... $(date)"
while kill -0 154709 2>/dev/null; do
    sleep 30
    echo "[condition_eval] Still waiting... $(date)"
done

echo "[condition_eval] Training done. Starting per-condition evaluation at $(date)"
cd /nfsshare/joseva
python yolact_condition_eval.py > /nfsshare/joseva/condition_eval_log.txt 2>&1
echo "[condition_eval] Evaluation complete at $(date)"
