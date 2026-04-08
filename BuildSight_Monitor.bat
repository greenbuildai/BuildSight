@echo off
setlocal enabledelayedexpansion
title BuildSight Phase 2 — LIVE PIPELINE MONITOR
color 0A

:loop
cls
echo ============================================================
echo   BuildSight Phase 2 — LIVE MONITOR    [%date% %time:~0,8%]
echo ============================================================
echo.

REM ── EVAL GRID (SASTRA) ─────────────────────────────────────
echo [SASTRA NODE1 — EVAL GRID PROGRESS]
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no joseva@172.16.13.62 "ssh node1 \"echo 'Jobs done:' && ls /nfsshare/joseva/condition_eval_results/*.json 2>/dev/null | grep -v matrix | wc -l && echo '/16 jobs' && echo '' && echo 'Last 5 log lines:' && tail -5 /nfsshare/joseva/logs/master_eval.log 2>/dev/null\"" 2>nul
echo.

REM ── ANNOTATION COUNTS ──────────────────────────────────────
echo [ANNOTATION COUNTS — target: 888 each]
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no joseva@172.16.13.62 "ssh node1 \"for m in YOLOv11 YOLOv26 YOLACT_plusplus SAMURAI_GT; do cnt=\$(find /nfsshare/joseva/val_annotated/\$m -maxdepth 1 -type f 2>/dev/null | wc -l); printf '  %-24s %s\n' \$m \$cnt; done\"" 2>nul
echo.

REM ── GPU STATUS ─────────────────────────────────────────────
echo [GPU STATUS — node1 A100]
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no joseva@172.16.13.62 "ssh node1 \"nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{printf \"  GPU%s: %s%% util | %s/%s MiB VRAM\n\", \$1, \$2, \$3, \$4}'\"" 2>nul
echo.

REM ── LOCAL DOCS STATUS ──────────────────────────────────────
echo [DOCS — comparative_study.md]
python -c "
import os
p='e:/Company/Green Build AI/Prototypes/BuildSight/docs/comparative_study.md'
if os.path.exists(p):
    t=open(p,encoding='utf-8').read()
    lines=t.splitlines()
    pending=sum(1 for l in lines if any(x in l for x in ['pending','Will be updated','[TODO]']))
    has57='5.7' in t
    hasDog=('dog' in t.lower() and 'false positive' in t.lower())
    hasSam=('tracker' in t.lower() and 'samurai' in t.lower())
    print(f'  Size: {round(os.path.getsize(p)/1024,1)} KB  |  Lines: {len(lines)}')
    print(f'  Pending markers: {pending}  |  Sec5.7: {\"YES\" if has57 else \"NO\"}  DogFP: {\"YES\" if hasDog else \"NO\"}  SAMURAI-note: {\"YES\" if hasSam else \"NO\"}')
    if pending==0: print('  STATUS: CLEAN - No pending markers')
    else: print(f'  STATUS: {pending} sections still need attention')
else:
    print('  File not found')
" 2>nul
echo.

REM ── MATRIX FILE ────────────────────────────────────────────
echo [condition_eval_matrix.json]
python -c "
import json, os
p='e:/Company/Green Build AI/Prototypes/BuildSight/docs/condition_eval_matrix.json'
if os.path.exists(p):
    d=json.load(open(p))
    nulls=sum(1 for m in d.values() if isinstance(m,dict) for v in m.values() if isinstance(v,dict) and v.get('mAP50') is None)
    total=sum(1 for m in d.values() if isinstance(m,dict) for v in m.values() if isinstance(v,dict))
    print(f'  Entries: {total} total | {nulls} null | {total-nulls} filled')
    if d.get('summary'): print(f'  Winner: {d[\"summary\"].get(\"winner\",\"?\")} | YOLOv11 mean mAP50: {d[\"summary\"][\"mean_mAP50\"].get(\"yolo11\",\"?\")}')
else:
    print('  File not found')
" 2>nul
echo.

echo ============================================================
echo   Auto-refresh in 30s — Press Ctrl+C to stop
echo ============================================================
timeout /t 30 /nobreak >nul
goto loop
