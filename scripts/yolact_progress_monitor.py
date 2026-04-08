import time, os, re, urllib.request
from datetime import datetime

LOG = os.path.expanduser("~/tournament_log.txt")
TOPIC = "https://ntfy.sh/buildsight-tournament-2026"
MAX_ITER = 80000
NOTIFY_EVERY = 2000   # notify every 2000 iters (~11 min)
CHECK_INTERVAL = 60   # check log every 60 seconds

def send(title, msg, tags="chart_with_upwards_trend"):
    try:
        t = title.encode("ascii", "ignore").decode("ascii")
        urllib.request.urlopen(
            urllib.request.Request(
                TOPIC, data=msg.encode("utf-8"),
                headers={"Title": t, "Tags": tags}
            ), timeout=10)
        print("[{}] Sent: {}".format(datetime.now().strftime("%H:%M:%S"), t))
    except Exception as e:
        print("Send error: {}".format(e))

def parse_log(path):
    if not os.path.exists(path):
        return None, None, None
    try:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
        m = re.findall(
            r"\[\s*\d+\]\s+(\d+)\s+\|\|.*?T:\s+([\d.]+)\s+\|\|\s+ETA:\s+([^\|]+)\|\|",
            raw
        )
        if not m:
            return None, None, None
        last = m[-1]
        return int(last[0]), float(last[1]), last[2].strip()
    except:
        return None, None, None

last_notified = -1
start_str = datetime.now().strftime("%b %d %H:%M")
print("YOLACT++ Progress Monitor running.")
print("Max iters: {}  |  Notifying every {} iters".format(MAX_ITER, NOTIFY_EVERY))
print("ntfy topic: " + TOPIC)

while True:
    iteration, loss, eta = parse_log(LOG)

    if iteration is not None:
        pct = round((iteration / MAX_ITER) * 100, 2)
        remaining = MAX_ITER - iteration
        milestone = (iteration // NOTIFY_EVERY) * NOTIFY_EVERY

        if milestone > last_notified and milestone > 0:
            last_notified = milestone
            filled = int(pct / 5)
            bar = "[" + "#" * filled + "-" * (20 - filled) + "]"
            title = "YOLACT++ {:.1f}% done ({}/{})".format(pct, iteration, MAX_ITER)
            msg = (
                "Progress: {} {:.1f}%\n"
                "Iter: {:,} / {:,} ({:,} left)\n"
                "Total Loss: {:.3f}\n"
                "ETA: {}\n"
                "Started: {}"
            ).format(
                bar, pct,
                iteration, MAX_ITER, remaining,
                loss, eta, start_str
            )
            send(title, msg)

        print("[{}] {:.2f}% | iter {:,} | loss {:.3f} | ETA {}".format(
            datetime.now().strftime("%H:%M:%S"), pct, iteration, loss, eta))
    else:
        print("[{}] Waiting for training iterations in log...".format(
            datetime.now().strftime("%H:%M:%S")))

    # Check for completion
    if os.path.exists(LOG):
        with open(LOG, "rb") as f:
            c = f.read().decode("utf-8", errors="ignore")
        if "ALL TRAINING COMPLETE" in c:
            send(
                "YOLACT++ Training COMPLETE!",
                "All {:,} iterations done!\nModel saved. Ready for evaluation.\nStarted: {}".format(
                    MAX_ITER, start_str),
                "tada"
            )
            print("Training complete! Exiting monitor.")
            break

    time.sleep(CHECK_INTERVAL)
