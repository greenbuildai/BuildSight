"""
SASTRA Supercomputer - Resumable Dataset Upload
Uses SFTP to upload only files that are missing or incomplete on the server.
Acts like rsync — skips files already fully transferred.
"""
import os
import sys
import time
import paramiko
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
GATEWAY    = "172.16.13.62"
USER       = "joseva"
PASSWORD   = "sastra123"
LOCAL_DIR  = r"D:\Jovi\Projects\BuildSight\Backup_Dataset(annotated)"
REMOTE_DIR = "/nfsshare/joseva/buildsight/dataset/Backup_Dataset(annotated)"
# ───────────────────────────────────────────────────────────────

def connect_sftp():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(GATEWAY, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()
    return client, sftp

def remote_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except FileNotFoundError:
        return False

def ensure_remote_dir(sftp, remote_path):
    parts = remote_path.replace("\\", "/").split("/")
    current = ""
    for part in parts:
        if not part:
            continue
        current += "/" + part
        try:
            sftp.stat(current)
        except FileNotFoundError:
            try:
                sftp.mkdir(current)
            except:
                pass

def get_all_local_files(local_dir):
    """Return list of (local_abs_path, relative_path) for all files."""
    files = []
    base = Path(local_dir)
    for p in base.rglob("*"):
        if p.is_file():
            files.append((str(p), str(p.relative_to(base))))
    return files

def format_size(bytes_val):
    if bytes_val >= 1 << 30:
        return f"{bytes_val/(1<<30):.2f} GB"
    elif bytes_val >= 1 << 20:
        return f"{bytes_val/(1<<20):.1f} MB"
    return f"{bytes_val/(1<<10):.1f} KB"

def mini_bar(pct, width=30):
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)

def upload_file(sftp, local_path, remote_path, filename, idx, total, start_time, uploaded_bytes):
    local_size = os.path.getsize(local_path)
    # Check if already fully uploaded
    try:
        remote_stat = sftp.stat(remote_path)
        if remote_stat.st_size == local_size:
            return 0, True  # skipped
    except FileNotFoundError:
        pass

    sftp.put(local_path, remote_path)
    return local_size, False

def main():
    os.system("cls")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SASTRA — Resumable Dataset Upload (Smart Skip Mode)   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Local  : {LOCAL_DIR}")
    print(f"  Remote : {REMOTE_DIR}")
    print("  Status : Scanning local files...")

    all_files = get_all_local_files(LOCAL_DIR)
    total_files = len(all_files)
    total_bytes = sum(os.path.getsize(f[0]) for f in all_files)

    print(f"  Found  : {total_files} files | {format_size(total_bytes)} total")
    print("\n  Connecting to SASTRA...")

    try:
        client, sftp = connect_sftp()
        print("  ✅ Connected!\n")
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        sys.exit(1)

    start_time = time.time()
    uploaded_bytes = 0
    uploaded_files = 0
    skipped_files  = 0
    failed_files   = []

    for idx, (local_path, rel_path) in enumerate(all_files, 1):
        remote_path = (REMOTE_DIR + "/" + rel_path.replace("\\", "/"))
        remote_folder = remote_path.rsplit("/", 1)[0]

        # Make sure remote directory exists
        ensure_remote_dir(sftp, remote_folder)

        filename = os.path.basename(local_path)
        pct_files = idx / total_files * 100
        
        # Dual Bar Logic
        skipped_pct = (skipped_files / total_files) * 100
        uploaded_pct = (uploaded_files / total_files) * 100
        
        elapsed = time.time() - start_time
        rate_bps = uploaded_bytes / elapsed if elapsed > 3 else 0
        remaining_bytes = total_bytes - uploaded_bytes
        eta_mins = (remaining_bytes / rate_bps / 60) if rate_bps > 0 else 0

        # Header/Status area (this stays at the bottom or top)
        # We'll print a status line, then the file line.
        
        status_line = (
            f"\r  OVERALL: [{mini_bar(pct_files, 20)}] {pct_files:.1f}%  "
            f"| SKIP: [{mini_bar(skipped_pct, 20)}] {skipped_files} "
            f"| SENT: [{mini_bar(uploaded_pct, 20)}] {uploaded_files} "
            f"| ETA: ~{eta_mins:.1f}m    "
        )
        sys.stdout.write(status_line)
        sys.stdout.flush()

        try:
            size, skipped = upload_file(sftp, local_path, remote_path, filename, idx, total_files, start_time, uploaded_bytes)
            if skipped:
                skipped_files += 1
                # To see every file, we print above the carriage return line
                print(f"\n  [SKIP] {rel_path}")
            else:
                uploaded_bytes += size
                uploaded_files += 1
                rate_mb = (uploaded_bytes / (1<<20)) / max((time.time()-start_time)/60, 0.001)
                print(f"\n  [SENT] {rel_path} ({format_size(size)}) | {rate_mb:.1f} MB/min")
        except Exception as e:
            failed_files.append((rel_path, str(e)))
            print(f"\n  [FAIL] {rel_path} — {e}")

    sftp.close()
    client.close()

    elapsed = time.time() - start_time
    print(f"\n\n  ✅ DONE! Upload complete in {elapsed/60:.1f} minutes")
    print(f"  Uploaded : {uploaded_files} new files ({format_size(uploaded_bytes)})")
    print(f"  Skipped  : {skipped_files} files (already on server)")
    if failed_files:
        print(f"\n  ⚠️  {len(failed_files)} files failed:")
        for f, e in failed_files[:10]:
            print(f"     {f} — {e}")
    print("\n  🎉 Dataset is ready for training on the A100!")
    input("\n  Press Enter to exit...")

if __name__ == "__main__":
    main()
