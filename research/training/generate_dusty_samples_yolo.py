import os
import json
import vertexai
from PIL import Image
from vertexai.preview.vision_models import ImageGenerationModel

# Try to import Ultralytics for annotation
try:
    from ultralytics import YOLO
    has_yolo = True
except ImportError:
    has_yolo = False

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
    print("Requesting 4 images...")
    response1 = model.generate_images(prompt=PROMPT, number_of_images=4, aspect_ratio="1:1")
    print("Requesting 1 more image...")
    response2 = model.generate_images(prompt=PROMPT, number_of_images=1, aspect_ratio="1:1")
    
    for img in response1: images.append(img)
    for img in response2: images.append(img)
        
    for i, img in enumerate(images):
        img_path = os.path.join(OUTPUT_DIR, f"dusty_site_{i+1:03d}.jpg")
        img.save(img_path)
        print(f"Saved: {img_path}")
        
except Exception as e:
    print("Error during image generation:", e)

if not images:
    print("No images generated. Exiting.")
    import sys; sys.exit(1)

if has_yolo:
    print("\nStarting auto-annotation with YOLO PPE model...")
    try:
        yolo_model = YOLO(r"E:\Company\Green Build AI\Prototypes\BuildSight\buildsight_v0\PPE_Detection\Model\ppe.pt")
        # model classes (usually 0: hardhat, 1: mask, 2: NO-Hardhat, 3: NO-Mask, 4: NO-Safety Vest, 5: Person, 6: Safety Cone, 7: Safety Vest, 8: machinery, 9: vehicle)
        # We need mapping. We will just save all detections.
        
        for i in range(1, 6):
            img_path = os.path.join(OUTPUT_DIR, f"dusty_site_{i:03d}.jpg")
            if not os.path.exists(img_path): continue
            
            results = yolo_model(img_path, verbose=False)
            txt_path = os.path.join(OUTPUT_DIR, f"dusty_site_{i:03d}.txt")
            
            with open(txt_path, "w") as f:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0].item())
                    # YOLO format: cls cx cy w h
                    cx, cy, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            print(f"Annotated {os.path.basename(img_path)} with {len(results[0].boxes)} objects.")
    except Exception as e:
        print(f"Annotation failed: {e}")
else:
    print("\nUltralytics YOLO not found. Skipping auto-annotation.")

print("\nTask Complete!")
