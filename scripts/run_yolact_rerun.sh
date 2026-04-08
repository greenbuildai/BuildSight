#!/bin/bash
LOG=/nfsshare/joseva/logs/master_eval.log
PYTHON=/nfsshare/joseva/.conda/envs/buildsight/bin/python3
SCRIPT=/nfsshare/joseva/val_condition_eval.py
GENMATRIX=/nfsshare/joseva/generate_condition_matrix.py

echo "[$(date)] === YOLACT++ RERUN START ===" >> $LOG

for cond in S1_normal S2_dusty S3_low_light S4_crowded; do
    echo "[$(date)] Running yolact++ x $cond" >> $LOG
    $PYTHON $SCRIPT --model yolact --condition $cond >> $LOG 2>&1
    RC=$?
    echo "[$(date)] Done yolact++ x $cond exit=$RC" >> $LOG
done

echo "[$(date)] Regenerating condition matrix..." >> $LOG
$PYTHON $GENMATRIX >> $LOG 2>&1
echo "[$(date)] === YOLACT++ RERUN COMPLETE ===" >> $LOG
