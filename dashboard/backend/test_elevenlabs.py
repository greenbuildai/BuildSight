import os
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load .env from dashboard/backend/.env
dotenv_path = Path("e:/Company/Green Build AI/Prototypes/BuildSight/dashboard/backend/.env")
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("ELEVENLABS_API_KEY")
voice_id = os.getenv("ELEVENLABS_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")
model_id = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

print(f"Testing ElevenLabs API Key: {api_key[:6]}...{api_key[-4:] if api_key else 'None'}")
print(f"Voice ID: {voice_id}")
print(f"Model ID: {model_id}")

if not api_key:
    print("ERROR: ELEVENLABS_API_KEY not found in .env file.")
    exit(1)

url = f"https://api.elevenlabs.io/v1/user"
headers = {
    "xi-api-key": api_key
}

try:
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        user_data = response.json()
        print("SUCCESS: ElevenLabs API Key is VALID.")
        print(f"User Subscription: {user_data.get('subscription', {}).get('tier', 'unknown')}")
        print(f"Character Count: {user_data.get('subscription', {}).get('character_count', 0)} / {user_data.get('subscription', {}).get('character_limit', 0)}")
    else:
        print(f"FAILED: ElevenLabs API Key is INVALID or there's an issue.")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"ERROR: Could not connect to ElevenLabs API: {e}")
