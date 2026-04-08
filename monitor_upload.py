"""
SASTRA Supercomputer - Dataset Upload Monitor (Enhanced)
Shows per-folder upload status with progress indicators.
"""
import time
import os
import sys

GATEWAY = "172.16.13.62"
USER = "joseva"
PASSWORD = "sastra123"
REMOTE_BASE = "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)"

# Local folder info: name -> (local_files, local_size_mb)
FOLDERS = {
    "SAMURAI":          (9183,  3023.4),
    "YOLACT_plusplus":  (4728,  2992.1),
    "YOLOv11":          (9510,  2972.9),
    "YOLOv26":          (9510,  2972.9),
}
TOTAL_FILES = sum(v[0] for v in FOLDERS.values())
TOTAL_MB    = sum(v[1] for v in FOLDERS.values())

try:
    import paramiko
except ImportError:
    print("Installing paramiko...")
    os.system("pip install paramiko -q")
    import paramiko

def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(GATEWAY, username=USER, password=PASSWORD, timeout=15)
    return client

def ssh_cmd(client, cmd):
    _, out, _ = client.exec_command(cmd)
    return out.read().decode().strip()

def get_folder_stats(client):
    stats = {}
    for folder in FOLDERS:
        remote_path = f"{REMOTE_BASE}/{folder}"
        # File count
        count_raw = ssh_cmd(client, f"ssh node1 find '{remote_path}' -type f 2>/dev/null | wc -l")
        # Size in MB
        size_raw  = ssh_cmd(client, f"ssh node1 du -sm '{remote_path}' 2>/dev/null | cut -f1")
        try:
            count = int(count_raw)
        except:
            count = 0
        try:
            size_mb = float(size_raw)
        except:
            size_mb = 0.0
        stats[folder] = (count, size_mb)
    return stats

def mini_bar(pct, width=20):
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)

def status_icon(pct):
    if pct >= 99:   return "✅"
    if pct >= 50:   return "🔄"
    if pct > 0:     return "⬆️ "
    return              "⏳"

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    INTERVAL = 30
    start_time = time.time()

    while True:
        clear()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     SASTRA Supercomputer — Dataset Upload Monitor        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Server : {GATEWAY}  |  Path: ...buildsight/dataset/")
        print(f"  Time   : {time.strftime('%H:%M:%S')}  |  Refreshing every {INTERVAL}s")
        print("──────────────────────────────────────────────────────────")

        try:
            print("  Connecting to server...", end="\r")
            client = connect()
            stats = get_folder_stats(client)
            client.close()

            total_done_files = 0
            total_done_mb    = 0.0

            print("                                                          ")
            print("  📁 Backup_Dataset(annotated)/")

            for i, (folder, (local_files, local_mb)) in enumerate(FOLDERS.items()):
                done_files, done_mb = stats.get(folder, (0, 0.0))
                pct_files = min((done_files / local_files * 100) if local_files > 0 else 0, 100)
                pct_size  = min((done_mb    / local_mb    * 100) if local_mb    > 0 else 0, 100)
                pct = (pct_files + pct_size) / 2
                icon = status_icon(pct)
                bar  = mini_bar(pct, 18)
                connector = "└──" if i == len(FOLDERS) - 1 else "├──"
                print(f"  {connector} {icon} {folder}/")
                print(f"  │     [{bar}] {pct:.0f}%  |  {done_mb:.0f}/{local_mb:.0f} MB  |  {done_files}/{local_files} files")

                total_done_files += done_files
                total_done_mb    += done_mb

            # Overall
            overall_pct = min((total_done_mb / TOTAL_MB * 100) if TOTAL_MB > 0 else 0, 100)
            elapsed = time.time() - start_time
            rate_mb_per_min = (total_done_mb / (elapsed / 60)) if elapsed > 0 else 0
            remaining_mb = max(TOTAL_MB - total_done_mb, 0)
            eta_mins = (remaining_mb / rate_mb_per_min) if rate_mb_per_min > 0 else 0

            print("──────────────────────────────────────────────────────────")
            print(f"  Overall  : [{mini_bar(overall_pct, 30)}] {overall_pct:.1f}%")
            print(f"  Uploaded : {total_done_mb:.0f} MB / {TOTAL_MB:.0f} MB  ({total_done_files} / {TOTAL_FILES} files)")
            print(f"  Speed    : ~{rate_mb_per_min:.1f} MB/min")
            print(f"  ETA      : ~{eta_mins:.0f} minutes remaining")
            print("──────────────────────────────────────────────────────────")

            if overall_pct >= 99:
                print("\n  🎉 UPLOAD COMPLETE! All folders transferred to SASTRA!")
                print("  You can now start model training.")
                input("\n  Press Enter to exit...")
                break

        except Exception as e:
            print(f"\n  ⚠️  Connection error: {e}")
            print("     Retrying...")

        print(f"\n  Next update in {INTERVAL} seconds... (Ctrl+C to exit)")
        time.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Monitor stopped.")
