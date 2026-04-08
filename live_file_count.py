import paramiko
import time
import os

GATEWAY = "172.16.13.62"
USER = "joseva"
PASSWORD = "sastra123"

print("\n🚀 Connecting to SASTRA to monitor live file count...\n")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(GATEWAY, username=USER, password=PASSWORD, timeout=30)
    print("✅ Connected!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit(1)

TOTAL_FILES = 32931

def make_bar(pct, length=40):
    filled = int(length * pct / 100)
    return "█" * filled + "░" * (length - filled)

while True:
    try:
        # Run standard command to count files recursively directly on the NFS gateway
        stdin, stdout, stderr = client.exec_command("find /nfsshare/joseva/buildsight/dataset -type f | wc -l")
        
        output = stdout.read().decode().strip()
        
        # Parse the output
        if output.isdigit():
            count = int(output)
            pct = (count / TOTAL_FILES) * 100
            bar = make_bar(pct)
            print(f"\r  [{bar}] {pct:5.2f}%  |  📦 Uploaded: {count:,} / {TOTAL_FILES:,} files     ", end="", flush=True)
        else:
            print(f"\r  ⚠️ Counting... ({output.strip()})  {' '*30}", end="", flush=True)

        time.sleep(5)
    except Exception as e:
        print(f"\n⚠️ Error checking count: {e}")
        time.sleep(5)
