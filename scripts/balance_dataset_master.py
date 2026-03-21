import os
import json
import time
import random
import vertexai
import hashlib
from PIL import Image
from vertexai.preview.vision_models import ImageGenerationModel
from concurrent.futures import ThreadPoolExecutor

# Path to the JSON key
KEY_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\ecocraft-designer-470803-96ef945c1dc0.json"
BASE_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset"

TARGETS = {
    "Normal_Site_Condition": 1300,
    "Crowded_Condition": 1300,
    "Dusty_Condition": 1300,
    "Low_Light_Condition": 1300
}

# Distributed global endpoints for max speed
REGIONS = ["us-central1", "us-east4", "europe-west1", "asia-southeast1", "us-west1", "europe-west4"]

CATEGORIES = {
    "Normal_Site_Condition": {
        "prompts": [
            "A highly realistic photo of an active Indian construction site, massive concrete pillars, steel rebar, scaffolding, Indian construction workers in safety vests and hardhats, clear bright day, neutral white midday sunlight, 8k.",
            "A sharp photo of Indian construction workers on a large project, bright midday sunlight, blue sky, reddish-brown soil, rebar structures, workers in orange/yellow vests and hardhats, 8k.",
            "Realistic photo of massive scaffolding systems on an Indian construction site, bright blue sky, harsh midday sunlight, workers in safety vests, red-brown terrain, 8k.",
            "Deep foundation work at an Indian construction site, intense midday sun, red-brown soil, workers in safety vests assembling rebar, 8k.",
            "Interior of unfinished Indian commercial building, bright neutral midday daylight through windows, concrete floors, workers in hardhats and vests, 8k.",
            "Tower cranes operating on Indian construction site, bright blue sky, harsh midday sun, workers in safety gear, professional 8k photography."
        ],
        "target": TARGETS["Normal_Site_Condition"]
    },
    "Crowded_Condition": {
        "prompts": ["Very crowded busy Indian construction site, many workers in safety vests and hardhats working closely together, industrial atmosphere, red-brown terrain, 8k."],
        "target": TARGETS["Crowded_Condition"]
    },
    "Dusty_Condition": {
        "prompts": [
            "A highly realistic photo of Indian construction workers actively working INSIDE a localized cloud of white cement dust. The workers are partially obscured but their hardhats and safety vests are still detectable through the fine white dust particles. Natural daylight, active construction site, 8k resolution.",
            "A photorealistic photo of multiple Indian laborers in PPE working within a haze of fine stone-cutting dust on a construction site. The dust is concentrated around the workers, showing their forms clearly enough for detection while having fine dust particles suspended in the air. 8k resolution, realistic industrial environment.",
            "A realistic industrial photography of an Indian construction site where workers are standing and working amidst fine airborne soil dust during excavation. The workers are surrounded by localized dust haze but remain visible in the scene. Neutral white light, 8k resolution, no sandstorm.",
            "High quality photo of Indian construction workers in yellow hardhats and orange vests moving through a localized haze of white drywall and cement dust on an indoor-outdoor site. The dust surrounds the workers, but they remain detectable. 8k resolution."
        ],
        "target": TARGETS["Dusty_Condition"]
    },
    "Low_Light_Condition": {
        "prompts": [
            "A highly realistic photo of an active Indian construction site at night. Harsh industrial floodlights illuminating the scene, casting long shadows. Multiple Indian construction workers in reflective safety vests and hardhats are clearly visible working near scaffolding and rebar. Gritty atmospheric haze, 8k resolution.",
            "Photorealistic photo of an Indian construction site during evening dusk (blue hour). Dim natural lighting mixed with early artificial site lights. Indian laborers in high-visibility gear and hardhats working on structural formwork. Clear visibility of workers against the fading sky, 8k resolution.",
            "A realistic ground-level shot of an Indian construction project at night. Multiple workers in PPE (hardhats/vests) are detectable under the glow of mobile light towers. Background is dark with industrial structures partially visible. High-contrast lighting, 8k resolution.",
            "A highly realistic active Indian construction site at night with exposed rebar and unfinished structures. Harsh artificial floodlights partially obscured by a humid tropical haze and dust particles. Indian workers in high-visibility gear working inside shadows and haze. Gritty industrial night atmosphere, red-brown terrain, 8k resolution."
        ],
        "target": TARGETS["Low_Light_Condition"]
    }
}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH
with open(KEY_PATH, "r") as f:
    project_id = json.load(f)["project_id"]

def calculate_hash(filepath):
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception: return None

def worker_task(region):
    vertexai.init(project=project_id, location=region)
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
    
    while True:
        pending = []
        for cat, info in CATEGORIES.items():
            cat_path = os.path.join(BASE_DIR, cat)
            current = len([f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            if current < info["target"]:
                pending.append((cat, info, cat_path, current))
        
        if not pending: break
            
        cat_name, info, cat_path, current = random.choice(pending)
        prompt = random.choice(info["prompts"])
        
        try:
            response = model.generate_images(prompt=prompt, number_of_images=4, aspect_ratio="1:1")
            ts = int(time.time() * 1000)
            for i, image in enumerate(response):
                image.save(os.path.join(cat_path, f"syn_{cat_name.lower()}_{region}_{ts}_{i}.jpg"))
            print(f"[{region}] Generated 4 images for {cat_name} ({current+4}/{info['target']})")
            time.sleep(1)
        except Exception as e:
            if "429" in str(e): time.sleep(25)
            else: time.sleep(5)

def run_dataset_audit():
    print(f"\n🔍 STARTING DATASET AUDIT (DEADLINE 11PM): {BASE_DIR}")
    Stats = {'total': 0, 'duplicates': 0, 'low_res': 0, 'corrupted': 0, 'converted': 0}
    seen_hashes = set()
    
    for cat in CATEGORIES.keys():
        cat_path = os.path.join(BASE_DIR, cat)
        if not os.path.exists(cat_path): continue
        for file in os.listdir(cat_path):
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')): continue
            filepath = os.path.join(cat_path, file)
            Stats['total'] += 1
            fhash = calculate_hash(filepath)
            if not fhash or fhash in seen_hashes:
                os.remove(filepath); Stats['duplicates'] += 1; continue
            seen_hashes.add(fhash)
            try:
                with Image.open(filepath) as img:
                    img.verify()
                with Image.open(filepath) as img:
                    w, h = img.size
                    if w < 120 or h < 120: os.remove(filepath); Stats['low_res'] += 1; continue
                    if img.mode != 'RGB' or not file.lower().endswith('.jpg'):
                        rgb_img = img.convert('RGB')
                        new_name = os.path.splitext(filepath)[0] + ".jpg"
                        rgb_img.save(new_name, "JPEG", quality=95)
                        if filepath != new_name: os.remove(filepath)
                        Stats['converted'] += 1
            except Exception: os.remove(filepath); Stats['corrupted'] += 1
    print("\n=== AUDIT RESULTS ==="); [print(f" - {k.capitalize()}: {v}") for k,v in Stats.items()]

if __name__ == "__main__":
    print(f"🔥 TURBO MULTI-REGION MODE ENABLED.")
    with ThreadPoolExecutor(max_workers=len(REGIONS)) as executor:
        executor.map(worker_task, REGIONS)
    run_dataset_audit()
