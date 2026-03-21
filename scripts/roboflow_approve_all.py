import requests
import time

# =====================================================================
# ROBOFLOW BULK APPROVE SCRIPT
# =====================================================================
API_KEY = "xu5GOj5rUJSSWi44QYPu"
WORKSPACE = "josevas-workspace"
PROJECT_NAME = "buildsight-auto-labels"

def approve_all_batches():
    print(f"Connecting to Roboflow to automate the Approval phase...")
    
    # 1. Fetch all batches in the 'Annotate' tab
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_NAME}/batches?api_key={API_KEY}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching batches: {response.status_code}")
        return

    batches = response.json().get("batches", [])
    print(f"Found {len(batches)} batches awaiting approval.")

    for batch in batches:
        batch_id = batch.get("id")
        # We only want to approve batches that have unapproved images
        if batch.get("unannotated", 0) > 0 or batch.get("unapproved", 0) > 0:
            print(f"Triggering 'Accept All' for Batch: {batch_id}...")
            
            # The Roboflow API endpoint to 'Accept' a batch
            approve_url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_NAME}/batches/{batch_id}/approve?api_key={API_KEY}"
            approve_res = requests.post(approve_url)
            
            if approve_res.status_code == 200:
                print(f"Successfully approved Batch {batch_id}!")
            else:
                print(f"Could not approve Batch {batch_id}: {approve_res.text}")
            
            time.sleep(1) # Polite pause to prevent rate limits

    print("Approval automation sequence finalized.")

if __name__ == "__main__":
    approve_all_batches()
