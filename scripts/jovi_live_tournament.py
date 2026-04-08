import os
import time
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight")
TASK_STATE_PATH = PROJECT_ROOT / "TASK_STATE.json"

def show_tournament():
    while True:
        if not TASK_STATE_PATH.exists():
            print("Waiting for TASK_STATE.json...")
            time.sleep(5)
            continue
            
        with open(TASK_STATE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        os.system('cls' if os.name == 'nt' else 'clear')
        
        current = data.get("current_task", {})
        status = current.get("status", "Unknown")
        yolact = data.get("yolact_results", {})
        
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       BuildSight AI — MODEL TOURNAMENT LIVE RADAR          ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f" TIME: {datetime.now().strftime('%H:%M:%S')} | STATUS: {status[:40]}...")
        print("╟────────────────────────────────────────────────────────────╢")
        print(" 🏆 YOLACT++ EMPIRICAL RESULTS (SASTRA NODE1)")
        print("╟────────────────────────────────────────────────────────────╢")
        
        conditions = ["S1_normal", "S2_dusty", "S3_low_light", "S4_crowded"]
        print(f" {'Condition':<15} | {'mAP50':<8} | {'Recall':<8} | {'F1':<8} | {'Lat':<6}")
        print("-" * 58)
        
        for c in conditions:
            res = yolact.get(c, {})
            name = c.replace("_", " ").title()
            m50 = res.get("mAP50", 0)
            rec = res.get("R", 0)
            f1 = res.get("F1", 0)
            lat = res.get("lat_ms", 0)
            
            # Progress bar for mAP50
            bar_len = 10
            filled = int(bar_len * m50)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f" {name:<15} | {m50:<8.4f} | {rec:<8.4f} | {f1:<8.4f} | {lat:<4.1f}ms")
            
        print("╟────────────────────────────────────────────────────────────╢")
        print(" 🛠  LEON (CODEX) — NODE1 TASKS")
        print("╟────────────────────────────────────────────────────────────╢")
        leon = current.get("leon_tasks", {})
        for k, v in leon.items():
            icon = "✅" if v == "complete" else "⏳"
            print(f" {icon} {k:<20}: {v}")
            
        print("╟────────────────────────────────────────────────────────────╢")
        print(" [Ctrl+C to exit] | Monitoring for Phase 3 Ensemble trigger... ")
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        show_tournament()
    except KeyboardInterrupt:
        print("\nRadar terminated.")
