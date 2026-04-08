import os
import time
import json
from pathlib import Path
from datetime import datetime

# CONFIG
PROJECT_ROOT = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight")
LABEL_DIR    = PROJECT_ROOT / "Dataset" / "Final_Annotated_Dataset" / "labels"
# Track when THIS dashboard instance started to isolate current session rate
SESSION_START_TIME = time.time()

def load_task_state():
    state_path = PROJECT_ROOT / "TASK_STATE.json"
    if not state_path.exists():
        return "Unknown", 1000, PROJECT_ROOT / "Dataset" / "Indian Dataset" / "Normal_Site_Condition"
    
    with open(state_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    task = data.get("current_task", {})
    condition_name = task.get("condition", "Normal_Site_Condition")
    desc = task.get("description", "Annotation Run")
    
    src_dir = PROJECT_ROOT / "Dataset" / "Indian Dataset" / condition_name
    # Count images in source to get target
    target_total = len(list(src_dir.glob("*.jpg")))
    
    return desc, target_total, src_dir

DASHBOARD_REFRESH = 5 # seconds

def get_stats(src_dir):
    # Count current condition labels across ALL splits (train/val/test)
    target_stems = set(f.stem.lower() for f in src_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg'))
    # Use rglob to find .txt in all subdirectories (train/val/test)
    all_label_files = list(LABEL_DIR.rglob("*.txt"))
    
    current_count = 0
    latest_file = "None"
    latest_time = 0
    # first_time of the ENTIRE dataset for this condition
    global_first_time = float('inf')
    session_mtimes = []
    
    for f in all_label_files:
        if f.stem.lower() in target_stems:
            current_count += 1
            try:
                mtime = f.stat().st_mtime
                if mtime > latest_time:
                    latest_time = mtime
                    latest_file = f.name
                if mtime < global_first_time:
                    global_first_time = mtime
                # Only include files processed AFTER this dashboard started for 'Session' metrics
                if mtime > SESSION_START_TIME:
                    session_mtimes.append(mtime)
            except: continue
                
    if global_first_time == float('inf'):
        global_first_time = time.time()
                
    return current_count, latest_file, latest_time, global_first_time, session_mtimes

def run_dashboard():
    try:
        while True:
            desc, target_total, src_dir = load_task_state()
            os.system('cls' if os.name == 'nt' else 'clear')
            count, latest, ltime, global_start, session_mtimes = get_stats(src_dir)
            
            # SESSION METRICS (files processed since dashboard started)
            session_count = len(session_mtimes)
            session_elapsed = time.time() - SESSION_START_TIME
            
            # MOVING AVERAGE RATE (Last 15 files) — handles skip sequences and thermal throttling
            moving_rate = 0
            if len(session_mtimes) >= 2:
                session_mtimes.sort()
                window = 15
                recent_window = session_mtimes[-window:]
                if len(recent_window) >= 2:
                    window_duration = recent_window[-1] - recent_window[0]
                    moving_rate = window_duration / (len(recent_window) - 1)
            
            # If session is empty or just starting, use global rate as fallback
            if moving_rate <= 0:
                global_elapsed = time.time() - global_start if count > 0 else 0
                moving_rate = global_elapsed / count if count > 0 else 12.0 # fallback default
            
            # Cap rate to avoid unreal jitter
            rate_to_use = max(0.2, moving_rate)
            
            bar_len = 40
            progress = count / target_total if target_total > 0 else 1.0
            filled = int(bar_len * min(1.0, progress))
            bar = "█" * filled + "░" * (bar_len - filled)
            
            remaining = max(0, target_total - count)
            eta_sec = remaining * rate_to_use if rate_to_use > 0 else 0
            eta_min = eta_sec / 60
            
            print("╔════════════════════════════════════════════════════════════╗")
            print("║       BuildSight AI — LIVE ANNOTATION DASHBOARD            ║")
            print("╚════════════════════════════════════════════════════════════╝")
            print(f" TIME: {datetime.now().strftime('%H:%M:%S')}")
            print(f" TARGET: {desc}")
            print(f" MASTER START: {datetime.fromtimestamp(global_start).strftime('%Y-%m-%d %H:%M:%S')}")
            print(" 🔗 STATUS: ACTIVE & MONITORING (MULTI-SPLIT SYNC)")
            print("╟────────────────────────────────────────────────────────────╢")
            print(f" REAL PROGRESS: {count}/{target_total} ({ (count/target_total)*100:.1f}%)")
            print(f" PROGRESS BAR:  [{bar}]")
            print(f" SESSION:       {session_count} images processed since UI launch")
            print(f" LATEST:        {latest} (finished {int(time.time()-ltime)}s ago)")
            print("╟────────────────────────────────────────────────────────────╢")
            print(f" SESSION SPEED: {rate_to_use:.2f} s/img (Moving Avg)")
            print(f" GLOBAL RATE:   { (time.time()-global_start)/count if count > 0 else 0 :.2f} s/img")
            print(f" SESSION TIME:  {int(session_elapsed/60)}m {int(session_elapsed%60)}s")
            print(f" ETA:           {int(eta_min)}m {int(eta_sec%60)}s remaining")
            print("╟────────────────────────────────────────────────────────────╢")
            print(" [Ctrl+C to stop dashboard] | Auto-refreshing... ")
            
            time.sleep(DASHBOARD_REFRESH)
    except KeyboardInterrupt:
        print("\nDashboard closed.")

if __name__ == "__main__":
    run_dashboard()
