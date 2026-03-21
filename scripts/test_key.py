"""Quick test of the new API key with image classification."""
import sys
from google import genai
from PIL import Image
from io import BytesIO

API_KEY = "AIzaSyDwcWfK_dxLdRr2A-IaNTLjgpr_4fKaJ5w"
IMG_PATH = r"e:\Company\Green Build AI\Prototypes\BuildSight\Dataset\Indian Dataset\PPE_SASTRA_Dataset_3\DSC06033.JPG"

print("Testing API key...")
c = genai.Client(api_key=API_KEY)

print("Loading image...")
img = Image.open(IMG_PATH).convert("RGB")
buf = BytesIO()
img.save(buf, format="JPEG")
buf.seek(0)
img2 = Image.open(buf)

print("Calling gemini-2.0-flash-lite...")
try:
    r = c.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=["Classify: Normal_Site_Condition, Dusty_Condition, Low_Light_Condition, Crowded_Condition. Reply ONLY the name.", img2]
    )
    print(f"  Result: '{r.text}'")
    print(f"  Candidates: {r.candidates}")
except Exception as e:
    print(f"  ERROR (flash-lite): {e}")

print("\nCalling gemini-2.5-flash...")
try:
    buf.seek(0)
    img3 = Image.open(buf)
    r2 = c.models.generate_content(
        model="gemini-2.5-flash",
        contents=["Classify: Normal_Site_Condition, Dusty_Condition, Low_Light_Condition, Crowded_Condition. Reply ONLY the name.", img3]
    )
    print(f"  Result: '{r2.text}'")
    print(f"  Candidates: {r2.candidates}")
except Exception as e:
    print(f"  ERROR (2.5-flash): {e}")

print("\nDone.")
