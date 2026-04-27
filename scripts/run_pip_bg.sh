#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
#  BuildSight – Background Pip Installation Script
#  Launches Ultralytics install in the background via nohup
# ─────────────────────────────────────────────────────────────────────

# Ensure Conda is available
source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate

# Activate env
conda activate buildsight

# Kill any existing pip to avoid locking
pkill -9 pip || true

# Launch in background with logging
echo "🚀 Starting background installation of Ultralytics..."
nohup pip install ultralytics > /nfsshare/joseva/install_log.txt 2>&1 &

# Brief pause to verify PID
sleep 2
PID=$(pgrep -f "pip install ultralytics")

if [ -n "$PID" ]; then
    echo "✅ Background process started successfully! PID: $PID"
    echo "📂 Logs: /nfsshare/joseva/install_log.txt"
    echo "💡 Monitor with: tail -f ~/install_log.txt"
else
    echo "❌ Failed to start background process. Check ~/install_log.txt for errors."
fi
