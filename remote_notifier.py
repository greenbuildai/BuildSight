import time
import os
import re
import urllib.request
from datetime import datetime

LOG_FILE = os.path.expanduser("~/tournament_log.txt")
TOPIC = "https://ntfy.sh/buildsight-tournament-2026"

def send_notification(title, message):
    try:
        req = urllib.request.Request(
            TOPIC,
            data=message.encode('utf-8'),
            headers={
                "Title": title,
                "Tags": "rocket"
            }
        )
        urllib.request.urlopen(req, timeout=10)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Sent: {message}")
    except Exception as e:
        print(f"Failed to push: {e}")

def get_current_status():
    if not os.path.exists(LOG_FILE):
        return "Log file not found."

    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find the current model
    models_started = re.findall(r'\[\d/\d\]\s+(.*?)\s+Training START', content)
    current_model = models_started[-1].strip() if models_started else "Unknown Model"

    # Find the latest epoch based on YOLO format (e.g., 17/100)
    # The progress bar uses \r, but regex can still find the pattern
    yolo_epochs = re.findall(r'(\d+)/(\d+)\s+[\d\.]+G', content)
    
    # Find YOLACT++ format
    yolact_epochs = re.findall(r'\[Epoch (\d+)\]', content)

    if current_model == "YOLACT++" and yolact_epochs:
        return f"{yolact_epochs[-1]}th Epoch of YOLACT++"
    elif yolo_epochs:
        last_epoch, total = yolo_epochs[-1]
        return f"{last_epoch}th epoch of {current_model}"
    
    return f"Training {current_model} (Calculating epoch...)"

def check_completions(last_completed, silent=False):
    if not os.path.exists(LOG_FILE):
        return
    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    completions = re.findall(r'(.*?)\s+Training DONE', content)
    for model in completions:
        model = model.strip()
        if len(model) > 1 and model not in last_completed:
            if not silent:
                send_notification("Model Trained! 🎉", f"SUCCESS: {model} is fully trained!")
            last_completed.add(model)

if __name__ == "__main__":
    print("Starting Clean Log Watcher...")
    
    last_completed = set()
    check_completions(last_completed, silent=True) # Populate already finished models natively
    
    # 1. Immediate Smoke Test & Status
    msg = get_current_status()
    send_notification("BuildSight Watcher ACTIVE", f"SMOKE TEST: You will get clean updates now.\n\nRunning: {msg}")

    
    # 2. Continuous Watch Loop (Every 30 mins)
    while True:
        time.sleep(1800) # 30 minutes
        
        # Check if new model finished
        check_completions(last_completed)
        
        # Send latest epoch status
        msg = get_current_status()
        send_notification("Training Progress ⏳", msg)
