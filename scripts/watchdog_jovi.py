"""
watchdog_jovi.py
================
Monitors for competing annotation processes spawned by system Python (Jovi's processes)
and kills them immediately. Keeps running until the target annotation process completes.
"""
import time
import subprocess
import sys

JOVI_PYTHON = r"C:\Users\brigh\AppData\Local\Programs\Python\Python310\python.exe"
MY_VENV = r"E:\Company\Green Build AI\Prototypes\BuildSight\.venv\Scripts\python.exe"
ANNOTATION_SCRIPT = "annotate_indian_dataset.py"

def get_python_procs():
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-WmiObject Win32_Process -Filter \"name='python.exe'\" | "
         "Select-Object ProcessId,CommandLine | "
         "ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=10
    )
    import json
    try:
        data = json.loads(result.stdout.strip())
        if isinstance(data, dict):
            data = [data]
        return data or []
    except Exception:
        return []

def kill_pid(pid):
    subprocess.run(
        ["powershell", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
        capture_output=True, timeout=5
    )

def main():
    print(f"[Watchdog] Started. Monitoring for Jovi annotation processes every 3s.")
    my_annotation_alive_count = 0
    killed_total = 0

    while True:
        procs = get_python_procs()
        my_alive = False

        for p in procs:
            cmd = (p.get("CommandLine") or "")
            pid = p.get("ProcessId")
            if not cmd or not pid:
                continue

            is_my_proc = MY_VENV.lower() in cmd.lower() and ANNOTATION_SCRIPT in cmd
            is_jovi_proc = JOVI_PYTHON.lower() in cmd.lower() and ANNOTATION_SCRIPT in cmd

            if is_my_proc:
                my_alive = True

            if is_jovi_proc:
                print(f"[Watchdog] KILLING Jovi annotation process PID={pid}")
                kill_pid(pid)
                killed_total += 1

        if my_alive:
            my_annotation_alive_count += 1
            if my_annotation_alive_count % 20 == 1:
                print(f"[Watchdog] My annotation alive. Total Jovi kills: {killed_total}")
        else:
            if my_annotation_alive_count > 0:
                print(f"[Watchdog] My annotation process ended. Total Jovi kills: {killed_total}. Exiting.")
                break
            else:
                # Not started yet
                pass

        time.sleep(3)

if __name__ == "__main__":
    main()
