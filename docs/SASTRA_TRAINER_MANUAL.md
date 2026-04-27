# 🏁 BuildSight: SASTRA Supercomputer Training Manual 🚀

This guide provides everything needed to access the system and manage the **BuildSight AI** model training session.

## 🔑 1. Identity & Access
Contact the primary user for the **Password**.

*   **Gateway IP:** `172.16.13.62`
*   **Username:** `joseva`
*   **Target Node:** `node1` (A100 GPU Node)

### Step-by-Step Login:
1.  **Connect to Gateway:**
    ```bash
    ssh joseva@172.16.13.62
    ```
2.  **Jump to Compute Node:**
    ```bash
    ssh node1
    ```

---

## 🏗️ 2. Environment Activation
Once on **node1**, you MUST activate the environment before running anything:

1.  **Initialize Conda:**
    ```bash
    source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate
    ```
2.  **Activate BuildSight:**
    ```bash
    conda activate buildsight
    ```

---

## 🚉 3. Launch & Monitor Training

### Launch Training (Run this once):
```bash
python ~/train_buildsight.py
```

### Check Logs (If running in background):
If the session was started with `nohup`, use this to see live progress:
```bash
tail -f ~/install_log.txt  # (For setup logs)
tail -f ~/nohup.out        # (Standard background logs)
```

### Check GPU Health:
To see if the **A100 GPU** is working and check VRAM usage:
```bash
nvidia-smi
```

---

## 🛠️ 4. Common Troubleshooting
- **`conda: command not found`**: You forgot to run `source /SASTRA_GPFS_CLUSTER/apps/anaconda3/bin/activate`.
- **`client_loop: send disconnect`**: Your local internet blinked. Just log back in; the server is still running the task!
- **Permission Denied**: Ensure you are logged in as `joseva`.

---
*Created for BuildSight Model Training – SASTRA Supercomputer Cluster*
