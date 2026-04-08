import requests
import json
import os
import time
from pathlib import Path

SECRET = "B8veDdTUIjYiapqM9fJ6zbu5x1StccGwR"
URL = "https://jovi-claw-production.up.railway.app/relay"
PROJECT_ROOT = Path(r"e:\Company\Green Build AI\Prototypes\BuildSight")
TASK_STATE_PATH = PROJECT_ROOT / "TASK_STATE.json"

def get_phase2_stats():
    if not TASK_STATE_PATH.exists():
        return "Task State Unknown"
    
    with open(TASK_STATE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current = data.get("current_task", {})
    status = current.get("status", "Unknown")
    yolact = data.get("yolact_results", {})
    
    # Extract peak finding (Dusty S2)
    s2 = yolact.get("S2_dusty", {})
    s2_map = s2.get("mAP50", 0)
    s2_rec = s2.get("R", 0)
    
    # Extract Low-Light class stats
    s3 = yolact.get("S3_low_light", {})
    s3_per_class = s3.get("per_class", {})
    
    msg = (
        "🧠 *Jovi GeoAI Status Report* | PHASE 2\n"
        "---------------------------\n"
        "✅ *Model Tournament Documentation COMPLETE*\n"
        f"📊 *Status*: {status}\n\n"
        "*YOLACT++ PERFORMANCE HIGHLIGHTS:*\n"
        f"🌪 *Dusty (S2) Persistence*: mAP50={s2_map:.4f}, Recall={s2_rec:.4f}\n"
        f"🛡 *Safety Benchmark (S3/worker)*: AP50={s3_per_class.get('worker', 0):.4f}\n\n"
        "🛠 *Next Stage*: Phase 3 WBF Ensemble Evaluation\n"
        "---------------------------\n"
        "Phase 2 PhD Comparative Study finalized. Section 6.5 (The Dusty Paradox) authored."
    )
    return msg

def send_update(message):
    payload = {
        "secret": SECRET,
        "tool": "jovi_workspace_integration",
        "args": {
            "action": "send_wa_boss",
            "message": message
        }
    }
    try:
        response = requests.post(URL, json=payload, timeout=30)
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    status_msg = get_phase2_stats()
    print("Syncing with Jovi Claw...")
    if send_update(status_msg):
        print("Successfully synced with Telegram/WhatsApp!")
    else:
        print("Failed to reach Jovi Claw relay.")
