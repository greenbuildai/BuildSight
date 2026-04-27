import requests

API_KEY = "xu5GOj5rUJSSWi44QYPu"
WORKSPACE = "josevas-workspace"
PROJECT_NAME = "buildsight-auto-labels"

url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_NAME}?api_key={API_KEY}"
data = requests.get(url).json()
project = data.get("project", {})

total = project.get("images", 0)
unannotated = project.get("unannotated", 0)
annotated = total - unannotated

print(f"Total: {total}")
print(f"Annotated: {annotated}")
print(f"Unannotated: {unannotated}")
