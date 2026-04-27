import requests

# =====================================================================
# ROBOFLOW PROJECT DIAGNOSTIC SCRIPT
# =====================================================================
API_KEY = "xu5GOj5rUJSSWi44QYPu"
WORKSPACE = "josevas-workspace"
PROJECT_NAME = "buildsight-auto-labels"

def check_stats():
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_NAME}?api_key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        project = data.get("project", {})
        total = project.get("images", 0)
        unannotated = project.get("unannotated", 0)
        annotated = total - unannotated
        
        print("-" * 40)
        print(f"ROBOFLOW PROJECT: {PROJECT_NAME}")
        print(f"Total Images:     {total}")
        print(f"Annotated:        {annotated}")
        print(f"Unannotated:      {unannotated}")
        print(f"Sync Completion:  {(annotated/total)*100:.1f}%" if total > 0 else "0.0%")
        print("-" * 40)
    else:
        print(f"Error fetching stats: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    check_stats()
