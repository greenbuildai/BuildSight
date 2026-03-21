import os
import json
import torch
import vertexai
from PIL import Image
from vertexai.preview.vision_models import ImageGenerationModel

# GroundingDINO imports
import sys
sys.path.append(r"E:\Company\Green Build AI\Prototypes\BuildSight\GroundingDINO")
from groundingdino.util.inference import load_model, load_image, predict

PROMPT = (
    "A CCTV-style elevated camera view of an active construction site. "
    "Multiple Indian construction workers wearing safety helmets and safety vests are performing construction activities. "
    "Moderate construction dust (cement and soil dust) is present in the environment, forming a slight dust haze. "
    "The dust partially covers the workers, but their forms and PPE remain visually detectable. "
    "Realistic construction site photography, natural lighting, realistic dusty environment, no sandstorm, 1024x1024 resolution."
)

KEY_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\ecocraft-designer-470803-96ef945c1dc0.json"
OUTPUT_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian_Dataset\Dusty_Site_Condition"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

with open(KEY_PATH, "r") as f:
    project_id = json.load(f)["project_id"]

vertexai.init(project=project_id, location="us-central1")
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

print(f"Generating 5 images into {OUTPUT_DIR}...")
images = []
try:
    # Imagen limit is usually 4 per request
    print("Requesting first 4 images...")
    response1 = model.generate_images(prompt=PROMPT, number_of_images=4, aspect_ratio="1:1")
    print("Requesting 1 more image...")
    response2 = model.generate_images(prompt=PROMPT, number_of_images=1, aspect_ratio="1:1")
    
    for img in response1:
        images.append(img)
    for img in response2:
        images.append(img)
        
    for i, img in enumerate(images):
        img_path = os.path.join(OUTPUT_DIR, f"dusty_site_{i+1:03d}.jpg")
        img.save(img_path)
        print(f"Saved: {img_path}")
        
except Exception as e:
    print("Error during image generation:", e)

if not images:
    print("No images were generated. Exiting.")
    sys.exit(1)

print("\nStarting auto-annotation with GroundingDINO...")

DINO_CONFIG_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\weights\groundingdino_swint_ogc.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dino_model = load_model(DINO_CONFIG_PATH, DINO_WEIGHTS_PATH, device=device)

TEXT_PROMPT = "person . safety helmet . safety vest ."
CAT_MAP = {
    "person": 0,
    "safety helmet": 1,
    "safety vest": 2
}

for i in range(1, 6):
    img_path = os.path.join(OUTPUT_DIR, f"dusty_site_{i:03d}.jpg")
    if not os.path.exists(img_path):
        continue
        
    print(f"Annotating {img_path}...")
    image_source, image_tensor = load_image(img_path)
    
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=0.35,
        text_threshold=0.25,
        device=device
    )
    
    txt_path = os.path.join(OUTPUT_DIR, f"dusty_site_{i:03d}.txt")
    annotations_count = 0
    with open(txt_path, "w") as f:
        for idx in range(len(boxes)):
            phrase = phrases[idx]
            cat_id = None
            for key in CAT_MAP:
                if key in phrase:
                    cat_id = CAT_MAP[key]
                    break
            
            if cat_id is None:
                continue
                
            box = boxes[idx].cpu().numpy() # cx, cy, bw, bh (already normalized 0-1)
            cx, cy, bw, bh = box
            
            # YOLO format: class_id x_center y_center width height (space separated)
            f.write(f"{cat_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            annotations_count += 1
            
    print(f"  -> Saved {annotations_count} annotations to {os.path.basename(txt_path)}")

print("\nDone! Dataset generated and annotated.")
