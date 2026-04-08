from google import genai
from google.genai import types

api_key = "AIzaSyAozS3xFiIsqJ1wJi8WfqArxG_PcztxYQ8"
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents="Return a JSON object with keys 'W1', 'W2' and boolean values: {'W1': true, 'W2': false}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    print(f"SUCCESS. Response:\n{response.text}")
except Exception as e:
    print(f"ERROR: {e}")

