import time, os, re, subprocess, urllib.request
from datetime import datetime

LOG        = os.path.expanduser("~/tournament_log.txt")
BENCH_LOG  = os.path.expanduser("~/benchmark.log")
REPORT     = os.path.expanduser("~/MONDAY_REPORT.txt")
WEIGHTS_DIR= os.path.expanduser("~/yolact/weights")
EVAL_DIR   = "/nfsshare/joseva/condition_eval_results"
MASTER_LOG = "/nfsshare/joseva/logs/master_eval.log"
TOPIC      = "https://ntfy.sh/buildsight-tournament-2026"
YOLACT_MAX = 80000

# ─── ntfy sender ─────────────────────────────────────────────────────────────
def send(title, msg, tags="rocket"):
    try:
        t = title.encode("ascii", "ignore").decode("ascii")
        urllib.request.urlopen(
            urllib.request.Request(
                TOPIC, data=msg.encode("utf-8"),
                headers={"Title": t, "Tags": tags}
            ), timeout=10
        )
        print("[" + datetime.now().strftime("%H:%M:%S") + "] Sent: " + msg[:80])
    except Exception as e:
        print("Send Error: " + str(e))

# ─── helpers ──────────────────────────────────────────────────────────────────
def read_log(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

def gpu_status():
    try:
        r = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader"],
            timeout=5
        ).decode().strip()
        lines = []
        for row in r.splitlines():
            idx, name, used, total, util, temp = [x.strip() for x in row.split(",")]
            lines.append(name + " | " + used + "/" + total + " | " + util + " util | " + temp + "C")
        return "\n".join(lines)
    except:
        return "GPU info unavailable"

def running_procs():
    try:
        r = subprocess.check_output(
            ["ps", "-u", "joseva", "-f"], timeout=5
        ).decode()
        interesting = []
        keywords = ["train.py", "benchmark", "evaluate", "samurai", "yolact",
                    "val_condition_eval", "notifier", "python"]
        for line in r.splitlines()[1:]:
            if any(k in line.lower() for k in keywords) and "grep" not in line and "ps " not in line:
                parts = line.split()
                cmd = " ".join(parts[7:]) if len(parts) > 7 else line
                interesting.append(cmd[:70])
        return "\n".join(interesting) if interesting else "No active processes"
    except:
        return "Process info unavailable"

def latest_weights():
    try:
        files = sorted(
            [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth") and "buildsight" in f],
            key=lambda f: os.path.getmtime(os.path.join(WEIGHTS_DIR, f))
        )
        if files:
            f = files[-1]
            mt = datetime.fromtimestamp(os.path.getmtime(os.path.join(WEIGHTS_DIR, f)))
            return f + " (" + mt.strftime("%H:%M") + ")"
    except:
        pass
    return "No checkpoints yet"

def parse_yolact_progress(c):
    pat = (r'\[\s*\d+\]\s+(\d+)\s+\|\|\s+'
           r'B:\s*([\d.]+)\s+\|\s+C:\s*([\d.]+)\s+\|\s+M:\s*([\d.]+)'
           r'.*?T:\s*([\d.]+)\s+\|\|\s+ETA:\s*([^\|]+)\|\|')
    matches = re.findall(pat, c)
    if not matches:
        return None
    it, b, cls, m, tot, eta = matches[-1]
    cur = int(it)
    pct = round(cur / YOLACT_MAX * 100, 2)
    return {"iter": cur, "pct": pct,
            "b": float(b), "c": float(cls), "m": float(m),
            "total": float(tot), "eta": eta.strip()}

# ─── Phase 2: eval grid status ────────────────────────────────────────────────
def get_eval_grid_status():
    """Returns (done_count, total=16, list_of_done_jobs, is_complete)"""
    try:
        files = [f for f in os.listdir(EVAL_DIR)
                 if f.endswith(".json") and "matrix" not in f]
        done = len(files)
        jobs = [f.replace(".json","").replace("_S"," x S") for f in sorted(files)]
        is_complete = done >= 16
        return done, 16, jobs, is_complete
    except:
        return 0, 16, [], False

def get_eval_grid_msg(include_gpu=False):
    done, total, jobs, complete = get_eval_grid_status()
    pct = round(done / total * 100)
    bar_w = 16
    filled = int(bar_w * done / total)
    bar = "[" + "#" * filled + "-" * (bar_w - filled) + "]"
    models = {"yolo11": 0, "yolo26": 0, "yolact": 0, "samurai": 0}
    for j in jobs:
        for m in models:
            if m in j.lower():
                models[m] += 1
    breakdown = ""
    model_names = {"yolo11":"YOLOv11", "yolo26":"YOLOv26",
                   "yolact":"YOLACT++", "samurai":"SAMURAI"}
    for k, v in models.items():
        icon = "V" if v == 4 else (">" if v > 0 else ".")
        breakdown += icon + " " + model_names[k] + ": " + str(v) + "/4\n"

    status = "COMPLETE!" if complete else "Running..."
    msg = ("Phase 2 Eval Grid " + status + "\n"
           + bar + " " + str(pct) + "% (" + str(done) + "/16)\n\n"
           + breakdown)

    # Try last log line
    try:
        log = read_log(MASTER_LOG)
        last = [l for l in log.splitlines() if l.strip()]
        if last:
            msg += "\nLast: " + last[-1][-60:]
    except:
        pass

    if include_gpu:
        msg += "\n\nGPU: " + gpu_status()
    return msg

# ─── main status function ─────────────────────────────────────────────────────
def get_stat(include_gpu=False):
    gpu = ("\n\nGPU: " + gpu_status()) if include_gpu else ""

    # ── Phase 2: condition eval grid ──────────────────────────────────────────
    done, total, jobs, complete = get_eval_grid_status()
    if done > 0 or os.path.exists(EVAL_DIR):
        return get_eval_grid_msg(include_gpu=include_gpu)

    # ── Phase 1: tournament (legacy) ──────────────────────────────────────────
    report = read_log(REPORT)
    if report:
        if "ALL STAGES COMPLETE" in report:
            champ = re.findall(r"CHAMPION:\s+(.*?)\s*\n", report)
            runner = re.findall(r"RUNNER-UP:\s+(.*?)\s*\n", report)
            msg = "TOURNAMENT COMPLETE!\n"
            if champ:
                msg += "#1 Champion: " + champ[-1] + "\n"
            if runner:
                msg += "#2 Runner-up: " + runner[-1] + "\n"
            msg += "Ensemble is ready for deployment."
            return msg + gpu
        if "ENSEMBLE PIPELINE" in report:
            return "Building Final Multi-Model Ensemble..." + gpu
        if "RANKINGS" in report:
            return "Calculating Rankings & selecting TOP 2 models..." + gpu
        if "[4/4] SAMURAI" in report:
            return "Benchmarking SAMURAI (4/4)..." + gpu
        if "[3/4] YOLACT++" in report:
            return "Benchmarking YOLACT++ (3/4)..." + gpu
        if "[2/4] YOLOv26" in report:
            return "Benchmarking YOLOv26 (2/4)..." + gpu
        if "[1/4] YOLOv11" in report:
            return "Benchmarking YOLOv11 (1/4)..." + gpu
        return "Evaluation pipeline is running..." + gpu

    c = read_log(LOG)
    if "ALL TRAINING COMPLETE" in c:
        procs = running_procs()
        if "benchmark" in procs.lower():
            bench = read_log(BENCH_LOG)
            last = [l for l in bench.splitlines() if l.strip()]
            last_line = last[-1] if last else "starting..."
            return "Benchmarking all models!\nStatus: " + last_line + gpu
        return ("All training DONE! Waiting to start benchmarking.\n"
                "Models trained: YOLOv11, YOLOv26, YOLACT++, SAMURAI\n"
                "Latest checkpoint: " + latest_weights() + gpu)
    if "YOLACT++ Training START" in c and "YOLACT++ Training DONE" not in c:
        p = parse_yolact_progress(c)
        if p:
            filled = int(p["pct"] / 5)
            bar = "#" * filled + "-" * (20 - filled)
            return ("YOLACT++ on A100\n"
                    "[" + bar + "] " + str(p["pct"]) + "%\n"
                    "Iter: " + str(p["iter"]) + " / " + str(YOLACT_MAX) + "\n"
                    "Loss: " + str(round(p["total"], 3)) +
                    "  Box:" + str(round(p["b"], 2)) +
                    "  Cls:" + str(round(p["c"], 2)) +
                    "  Mask:" + str(round(p["m"], 2)) + "\n"
                    "ETA: " + p["eta"] + "\n"
                    "Checkpoint: " + latest_weights() + gpu)
        return "YOLACT++ warming up on A100..." + gpu
    if "YOLOv26 Training START" in c and "YOLOv26 Training DONE" not in c:
        y = re.findall(r"(\d+)/(\d+)\s+[\d.]+G", c)
        if y:
            done2, total2 = y[-1]
            pct = round(int(done2) / int(total2) * 100, 1)
            return "YOLOv26 [2/3]: " + str(pct) + "% (" + done2 + "/" + total2 + " epochs)" + gpu
        return "YOLOv26 training started..." + gpu
    if "YOLOv11 Training START" in c and "YOLOv11 Training DONE" not in c:
        y = re.findall(r"(\d+)/(\d+)\s+[\d.]+G", c)
        if y:
            done2, total2 = y[-1]
            pct = round(int(done2) / int(total2) * 100, 1)
            return "YOLOv11 [1/3]: " + str(pct) + "% (" + done2 + "/" + total2 + " epochs)" + gpu
        return "YOLOv11 training started..." + gpu
    return "Initializing tournament pipeline..." + gpu

def chk_done(last_completed, silent=False):
    c = read_log(LOG)
    for m in [x.strip() for x in re.findall(r"(.*?)\s+Training DONE", c)]:
        if len(m) > 1 and m not in last_completed:
            if not silent:
                send("Model Trained!", "SUCCESS: " + m + " is fully trained!\n" +
                     "Next: " + get_stat(), tags="white_check_mark")
            last_completed.add(m)

# ─── main loop ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    last_completed = set()
    last_eval_done = 0
    chk_done(last_completed, silent=True)
    send("BuildSight Watcher v2", "Online! Current status:\n" + get_stat(include_gpu=True))

    while True:
        # Phase 2: track eval grid milestones
        done, total, jobs, complete = get_eval_grid_status()
        if done > last_eval_done:
            tag = "tada" if complete else "bar_chart"
            title = ("Phase 2 COMPLETE!" if complete
                     else "Eval Job " + str(done) + "/" + str(total) + " done")
            send(title, get_eval_grid_msg(include_gpu=True), tags=tag)
            last_eval_done = done
            if complete:
                break

        c = read_log(LOG)
        is_yolact = ("YOLACT++ Training START" in c and
                     "YOLACT++ Training DONE" not in c)
        interval = 300 if done > 0 else (900 if is_yolact else 1800)
        time.sleep(interval)
        chk_done(last_completed)
        if done == 0:
            send("Training Progress", get_stat(include_gpu=True))
