import urllib.request, json, time, os
import remote_notifier_v2 as rn

CONTROL_TOPIC  = "https://ntfy.sh/buildsight-control-2026/json"
REPLY_TOPIC    = "https://ntfy.sh/buildsight-tournament-2026"
LOG            = os.path.expanduser("~/tournament_log.txt")

def reply(title, msg, tags="satellite"):
    try:
        t = title.encode("ascii", "ignore").decode("ascii")
        urllib.request.urlopen(
            urllib.request.Request(
                REPLY_TOPIC, data=msg.encode("utf-8"),
                headers={"Title": t, "Tags": tags}
            ), timeout=10
        )
        print("[reply] " + msg[:60])
    except Exception as e:
        print("Reply error: " + str(e))

def handle(cmd):
    cmd = cmd.lower().strip()
    print("CMD: " + cmd)

    if cmd == "status":
        reply("Status", rn.get_stat(include_gpu=True), tags="bar_chart")

    elif cmd in ("eval", "grid", "phase2"):
        reply("Phase 2 Eval Grid", rn.get_eval_grid_msg(include_gpu=True), tags="microscope")

    elif cmd == "matrix":
        # Read the condition_eval_matrix.json and summarise
        try:
            import json as js
            p = rn.EVAL_DIR + "/condition_eval_matrix.json"
            d = js.load(open(p))
            lines = ["Condition Matrix Summary:"]
            if "summary" in d:
                s = d["summary"]
                best = s.get("best_per_condition", {})
                for cond, info in best.items():
                    lines.append(cond + ": " + info.get("model","?") +
                                 " mAP50=" + str(info.get("mAP50","?")))
                lines.append("")
                for m, v in s.get("mean_mAP50", {}).items():
                    lines.append(m + " mean: " + str(v))
                lines.append("Winner: " + s.get("winner","?"))
            else:
                lines.append("Matrix data not yet available.")
            reply("Condition Matrix", "\n".join(lines), tags="trophy")
        except Exception as e:
            reply("Matrix", "Could not read matrix: " + str(e))

    elif cmd == "gpu":
        reply("GPU Status", rn.gpu_status(), tags="zap")

    elif cmd == "log":
        try:
            with open(rn.MASTER_LOG, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l for l in f.read().splitlines() if l.strip()]
            last = "\n".join(lines[-8:])
            reply("Last 8 Log Lines", last, tags="scroll")
        except:
            try:
                with open(LOG, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [l for l in f.read().splitlines() if l.strip()]
                last = "\n".join(lines[-8:])
                reply("Last 8 Log Lines", last, tags="scroll")
            except:
                reply("Log", "Could not read log file.")

    elif cmd == "eta":
        done, total, jobs, complete = rn.get_eval_grid_status()
        if done > 0 and not complete:
            remaining = total - done
            secs_per_job = 35
            mins = (remaining * secs_per_job) // 60
            reply("ETA",
                  "Eval Grid ETA\n"
                  "Done: " + str(done) + "/" + str(total) + "\n"
                  "Remaining: " + str(remaining) + " jobs\n"
                  "ETA: ~" + str(mins) + " min",
                  tags="hourglass_flowing_sand")
        elif complete:
            reply("ETA", "Grid is already complete! 16/16 done.", tags="tada")
        else:
            c = rn.read_log(LOG)
            p = rn.parse_yolact_progress(c)
            if p:
                remaining = rn.YOLACT_MAX - p["iter"]
                reply("ETA",
                      "YOLACT++ ETA\n"
                      "Done: " + str(p["iter"]) + "/" + str(rn.YOLACT_MAX) +
                      " (" + str(p["pct"]) + "%)\n"
                      "ETA: " + p["eta"],
                      tags="hourglass_flowing_sand")
            else:
                reply("ETA", "No active task.\n" + rn.get_stat())

    elif cmd == "ps" or cmd == "proc":
        reply("Running Processes", rn.running_procs(), tags="computer")

    elif cmd == "loss":
        c = rn.read_log(LOG)
        p = rn.parse_yolact_progress(c)
        if p:
            reply("Current Losses",
                  "Iter " + str(p["iter"]) + "\n"
                  "Total : " + str(round(p["total"], 4)) + "\n"
                  "Box   : " + str(round(p["b"], 4)) + "\n"
                  "Class : " + str(round(p["c"], 4)) + "\n"
                  "Mask  : " + str(round(p["m"], 4)),
                  tags="chart_with_downwards_trend")
        else:
            reply("Loss", "YOLACT++ not training.\n" + rn.get_stat())

    elif cmd == "checkpoint" or cmd == "ckpt":
        reply("Latest Checkpoint", rn.latest_weights(), tags="floppy_disk")

    elif cmd == "help":
        reply("Commands",
              "status   - full progress + GPU\n"
              "eval     - Phase 2 eval grid status\n"
              "matrix   - condition matrix summary\n"
              "gpu      - GPU VRAM & temp\n"
              "log      - last 8 log lines\n"
              "eta      - time remaining\n"
              "loss     - current loss breakdown\n"
              "ckpt     - latest checkpoint file\n"
              "ps       - running processes\n"
              "help     - this menu",
              tags="information_source")

    else:
        reply("Unknown Command",
              "Unknown: '" + cmd + "'\nSend 'help' for available commands.",
              tags="question")

def listen():
    print("[" + time.strftime("%H:%M:%S") + "] Listening on buildsight-control-2026...")
    try:
        with urllib.request.urlopen(
            urllib.request.Request(CONTROL_TOPIC), timeout=86400
        ) as response:
            for line in response:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if data.get("event") == "message":
                            msg = data.get("message", "").strip()
                            if msg:
                                handle(msg)
                    except:
                        pass
    except Exception as e:
        print("Listener disconnected: " + str(e))
        time.sleep(5)

if __name__ == "__main__":
    while True:
        listen()
