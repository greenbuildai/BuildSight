#!/bin/bash
LOG=/nfsshare/joseva/logs/master_eval.log
MODELS="yolo11 yolo26 yolact samurai"
CONDITIONS="S1_normal S2_dusty S3_low_light S4_crowded"
DONE=0
TOTAL=16
echo "[$(date)] [7/7] Starting 16-job eval grid" >> $LOG
for model in $MODELS; do
  for cond in $CONDITIONS; do
    DONE=$((DONE+1))
    echo "[$(date)] Running job $DONE/$TOTAL: $model x $cond" >> $LOG
    /nfsshare/joseva/.conda/envs/buildsight/bin/python3 /nfsshare/joseva/val_condition_eval.py --model $model --condition $cond >> $LOG 2>&1
    RC=$?
    echo "[$(date)] Finished job $DONE/$TOTAL: $model x $cond (exit=$RC)" >> $LOG
  done
done
echo "[$(date)] All 16 jobs done. Generating matrix..." >> $LOG
/nfsshare/joseva/.conda/envs/buildsight/bin/python3 /nfsshare/joseva/generate_condition_matrix.py >> $LOG 2>&1
echo "[$(date)] [7/7] PIPELINE COMPLETE" >> $LOG
