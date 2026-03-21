import os
import json
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def test_imagen():
    key_path = r"E:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\ecocraft-designer-470803-96ef945c1dc0.json"
    
    if not os.path.exists(key_path):
        print(f"Error: JSON file not found at {key_path}")
        return

    # Set authentication
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    # Read project ID
    with open(key_path, "r") as f:
        credentials = json.load(f)
        project_id = credentials.get("project_id")
        
    if not project_id:
        print("Error: Could not find project_id in JSON file.")
        return
        
    print(f"Authenticating with project: {project_id}...")
    vertexai.init(project=project_id, location="us-central1")

    # Try Imagen 4.0 (as shown in user's screenshot) or fallback to 3.0
    model_id = "imagen-3.0-generate-001" # safer fallback. Will try 4 first
    print("Initializing Image Generation Model...")
    
    try:
        model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        print(f"Successfully loaded {model_id}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate a test image
    prompt = "A highly realistic photo of a construction site under heavily dusty conditions. Huge dust clouds, airborne particulate matter, hazy atmosphere, poor visibility, workers wearing hardhats and safety vests in a dusty environment, natural lighting"
    
    print("Generating test image...")
    try:
        response = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1"
        )
        output_path = "test_imagen_dusty.jpg"
        response[0].save(output_path)
        print(f"Success! Image saved to {output_path}")
    except Exception as e:
        print(f"Error during image generation: {e}")

if __name__ == "__main__":
    test_imagen()
