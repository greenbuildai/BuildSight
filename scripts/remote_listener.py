import urllib.request
import json
import time

try:
    import remote_notifier as rn
except ImportError:
    print("Error: remote_notifier.py must be in the same folder!")
    exit(1)

TOPIC = "https://ntfy.sh/buildsight-control-2026/json"

def listen():
    print(f"[{time.strftime('%H:%M:%S')}] Active! Text 'status' to ntfy.sh/buildsight-control-2026")
    try:
        # Open an infinite streaming connection
        req = urllib.request.Request(TOPIC)
        with urllib.request.urlopen(req, timeout=86400) as response:  # 24h timeout
            for line in response:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    if data.get("event") == "message":
                        msg = data.get("message", "").lower().strip()
                        print(f"Received text: {msg}")
                        
                        if msg == "status":
                            print("Status requested! Replying...")
                            rn.send("On-Demand Status 👀", f"You asked for it! Running: {rn.get_stat()}")
                except Exception as parse_e:
                    print("Parse error:", parse_e)
    except Exception as e:
        print("Listener disconnected:", e)
        time.sleep(5) # Delay before crash loop

if __name__ == "__main__":
    while True:
        listen()  # Keep reviving if the stream drops
