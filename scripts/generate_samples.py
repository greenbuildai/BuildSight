import os
import json
import time
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

KEY_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\ecocraft-designer-470803-96ef945c1dc0.json"
OUTPUT_DIR = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Original specific prompt
PROMPT = "A highly realistic, photorealistic photo of an active Indian construction site during a clear, bright day. Massive concrete pillars, steel rebar, and scaffolding are visible. Multiple Indian construction workers in yellow, green, and orange safety vests and hardhats are actively working. The environment is clear and authentic with red-brown terrain and blue skies, 8k resolution."

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

# Project Setup
with open(KEY_PATH, "r") as f:
    project_id = json.load(f)["project_id"]

vertexai.init(project=project_id, location="us-central1")
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

def generate_samples():
    print(f"Generating 5 samples using the original prompt...")
    count = 0
    batches = [4, 1]
    
    for batch_size in batches:
        try:
            print(f"Generating batch of {batch_size}...")
            response = model.generate_images(
                prompt=PROMPT,
                number_of_images=batch_size,
                aspect_ratio="1:1"
            )
            
            for idx, image in enumerate(response):
                file_name = f"original_style_sample_{count}.jpg"
                file_path = os.path.join(OUTPUT_DIR, file_name)
                image.save(file_path)
                print(f"Saved: {file_path}")
                count += 1
                
            time.sleep(2) # Respect quota
                
        except Exception as e:
            print(f"API Error: {e}")

if __name__ == "__main__":
    generate_samples()
