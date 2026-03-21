import os
import json
import time
import random
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# Path to the JSON key
KEY_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\ecocraft-designer-470803-96ef945c1dc0.json"
DST_DUSTY = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

# Project Setup
with open(KEY_PATH, "r") as f:
    project_id = json.load(f)["project_id"]

vertexai.init(project=project_id, location="us-central1")
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

def generate_dusty_images():
    os.makedirs(DST_DUSTY, exist_ok=True)
    
    # Check existing count
    existing_dusty = len([f for f in os.listdir(DST_DUSTY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    target_count = 500
    images_to_generate = target_count - existing_dusty
    
    if images_to_generate <= 0:
        print(f"Target already met: {existing_dusty} images found.")
        return

    print(f"Current Dusty Images: {existing_dusty}")
    print(f"Target: {target_count}")
    print(f"Need to generate: {images_to_generate} images using Google Imagen 3.0")

    # A list of subtle prompt variations to guarantee diverse and unique images
    prompts = [
        "A highly realistic photo of a construction site under heavily dusty conditions. Huge dust clouds, airborne particulate matter, hazy atmosphere, poor visibility. Workers wearing hardhats and high-visibility safety vests working in a dusty environment, natural lighting.",
        "Photorealistic view of an active construction site covered in thick dust. Heavy machinery operating, kicking up massive dust clouds causing low visibility. Construction workers in PPE, hard hats, and vests navigating the hazy site.",
        "Wide angle shot of a dusty building construction site. The air is thick with hazardous dust and particulate matter, creating a hazy, sepia-toned atmosphere. Workers in safety gear, protective helmets, and reflective vests.",
        "A realistic ground-level photo of heavily dusty conditions at a large construction site. Plumes of dirt and dust blowing through scaffolding and concrete pillars. Workers equipped with hardhats and safety vests amidst the haze.",
        "High quality photography of a dusty construction zone. Airborne dust obscures the background, very hazy and gritty environment. Several workers wearing helmets and safety vests are visible through the thick dust cloud."
    ]

    generated_count = 0
    batch_size = 4 # Imagen can generate up to 4 images per API call

    start_offset = existing_dusty + 2000 # To prevent any filename conflicts
    
    while generated_count < images_to_generate:
        # Determine how many images we actually need this loop
        current_batch_size = min(batch_size, images_to_generate - generated_count)
        prompt = random.choice(prompts)
        
        try:
            print(f"Generating {current_batch_size} images... (Total generated so far: {generated_count}/{images_to_generate})")
            
            response = model.generate_images(
                prompt=prompt,
                number_of_images=current_batch_size,
                aspect_ratio="1:1"
            )
            
            for idx, image in enumerate(response):
                file_name = f"synthetic_imagen_dust_{start_offset + generated_count}.jpg"
                file_path = os.path.join(DST_DUSTY, file_name)
                image.save(file_path)
                generated_count += 1
                
            # Sleep slightly to respect Vertex AI quotas (usually 60 requests per minute)
            time.sleep(2)
            
        except Exception as e:
            print(f"API Error encountered: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    print(f"\nSuccessfully generated {generated_count} new dusty condition images using Google Imagen!")

if __name__ == "__main__":
    generate_dusty_images()
