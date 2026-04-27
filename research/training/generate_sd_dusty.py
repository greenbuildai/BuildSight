import os
import random
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

SRC_NORMAL = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Normal_Site_Condition"
DST_DUSTY = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\Dusty_Condition"

def main():
    os.makedirs(DST_DUSTY, exist_ok=True)
    
    # Check how many images we currently have in dusty directory
    existing_dusty = len([f for f in os.listdir(DST_DUSTY) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    target_count = 500
    images_to_generate = target_count - existing_dusty
    
    if images_to_generate <= 0:
        print(f"Target of {target_count} images already met or exceeded (current: {existing_dusty}).")
        return

    print(f"Current Dusty Images: {existing_dusty}")
    print(f"Targeting: {target_count} total Dusty images")
    print(f"Need to generate: {images_to_generate} images via Stable Diffusion")

    # Load images
    files = [f for f in os.listdir(SRC_NORMAL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    
    # Initialize Pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"\nLoading {model_id}...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to("cuda")

    prompt = "a highly realistic photo of a construction site under heavily dusty conditions. huge dust clouds, airborne particulate matter, hazy atmosphere, poor visibility, workers in a dusty environment, natural lighting"
    negative_prompt = "cartoon, 3d, rendering, blurry, overexposed, clean, clear sky, text, watermark"

    generated_count = 0
    start_offset = existing_dusty + 1000  # avoid conflicts
    
    print("\nStarting generations...")
    for i, file in enumerate(files):
        if generated_count >= images_to_generate:
            break
            
        src = os.path.join(SRC_NORMAL, file)
        try:
            init_img = Image.open(src).convert("RGB")
            # Resize properly for SD
            init_img = init_img.resize((512, 512))
            
            # Generate!
            with torch.autocast("cuda"):
                # strength = 0.5 retains 50% of the original structure
                output = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=init_img,
                    strength=0.55,
                    guidance_scale=7.5,
                    num_inference_steps=50
                ).images[0]
            
            dst = os.path.join(DST_DUSTY, f"synthetic_sd_dusty_{start_offset+i}_{file.split('.')[0]}.png")
            output.save(dst)
            generated_count += 1
            print(f"  Dusty SD Augmented: {generated_count}/{images_to_generate} ({dst})")
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    print(f"\n=== Pipeline Complete. Generated {generated_count} new dusty condition images ===")

if __name__ == "__main__":
    main()
