import os
import subprocess

# =====================================================================
# ROBOFLOW UPLOAD SCRIPT (CLI BATCH IMPORTER)
# =====================================================================
API_KEY = "xu5GOj5rUJSSWi44QYPu"
WORKSPACE = "josevas-workspace"
PROJECT_NAME = "buildsight-auto-labels"
DATASET_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Final_Annotated_Dataset"
CLI_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\.venv\Scripts\roboflow.exe"

def upload_dataset():
    print("Preparing secure environment for Roboflow CLI Bulk Import...")
    os.environ["ROBOFLOW_API_KEY"] = API_KEY
    
    cmd = [
        CLI_PATH,
        "import",
        DATASET_DIR,
        "-w", WORKSPACE,
        "-p", PROJECT_NAME,
        "-c", "20"  # 20 concurrent threads for fast upload
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("Dataset batch upload successfully completed via Roboflow CLI!")
    else:
        print("Upload threw an error. Please verify the Roboflow API keys and project names.")

if __name__ == "__main__":
    upload_dataset()
