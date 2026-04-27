import os
import json
import time
import random
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# Path to the JSON key
KEY_PATH = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\ecocraft-designer-470803-96ef945c1dc0.json"
DST_LOWLIGHT = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Low_Light_Condition"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = KEY_PATH

# Project Setup
with open(KEY_PATH, "r") as f:
    project_id = json.load(f)["project_id"]

vertexai.init(project=project_id, location="us-central1")
model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

def generate_lowlight_visible_workers():
    os.makedirs(DST_LOWLIGHT, exist_ok=True)
    
    # We want to add a specific number of these high-visibility samples
    # to improve the quality of the low-light detection for workers.
    images_to_generate = 150 
    
    print(f"Target: Generate {images_to_generate} high-visibility worker low-light images using Google Imagen 3.0")

    # Prompts focused on visible workers and PPE in dim/dark settings
    prompts = [
        "Highly realistic photo of an Indian construction site at dusk with very low light. Several Indian construction workers are visible, wearing yellow and white hard hats and high-visibility neon safety vests that catch the dim light. The workers are actively working with tools and materials. Gritty, cinematic low-light photography, 4k, photorealistic.",
        "Photorealistic view of a construction site in India during late evening, minimal artificial lighting. Indian workers in PPE are clearly visible in the foreground, silhouettes defined against the dark background. Their reflective safety vests are prominent. Moody atmosphere, realistic grain, high detail.",
        "Ground-level shot of a building site in India under low light conditions. The scene is dim but workers in hardhats and safety gear are clearly identifiable. One worker is near a dim work light, casting dramatic shadows. Realistic shadows, authentic site equipment, 4k resolution.",
        "A realistic photo of an Indian construction project at night with limited site lighting. Indian construction workers are seen navigating the site, their PPE (helmets and reflective vests) clearly visible even in the low light. The workers' faces and movements are discernable. No flashy filters, raw DSLR style.",
        "Indian construction workers in safety gear working under a single floodlight at night on a building site. High contrast, reflective stripes on vests glowing under the light. Very high detail, photorealistic construction environment.",
        "A grainy, realistic night-time photo of a construction area in India. Workers in bright orange and green safety vests are clearly visible through the darkness. The hardhats are bright and pick up minimal ambient light. Atmospheric, sharp focus on the workers."
    ]

    generated_count = 0
    batch_size = 4 # Imagen can generate up to 4 images per API call

    # Start offset to avoid existing synthetic naming if any
    start_offset = int(time.time())
    
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
                file_name = f"synthetic_lowlight_visible_{start_offset + generated_count}.jpg"
                file_path = os.path.join(DST_LOWLIGHT, file_name)
                image.save(file_path)
                generated_count += 1
                
            # Sleep slightly to respect Vertex AI quotas
            time.sleep(1.5)
            
        except Exception as e:
            print(f"API Error encountered: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    print(f"\nSuccessfully generated {generated_count} new low-light condition images with visible workers!")

if __name__ == "__main__":
    generate_lowlight_visible_workers()
