import asyncio
import os
import sys
import pyaudio
import traceback
from google import genai
from google.genai import types

os.environ['GEMINI_API_KEY'] = 'AIzaSyASZc7RXw_OG8OcMxzqk7o-mn9hR7uak1A'

# Audio defaults (Pyaudio)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000  # Gemini returns 24kHz PCM
SEND_SAMPLE_RATE = 16000     # We send 16kHz PCM
CHUNK = 512

p = pyaudio.PyAudio()

async def receive_from_gemini(session, speaker_stream):
    """Receive audio from Gemini and play it via PyAudio."""
    try:
        async for response in session.receive():
            if hasattr(response, 'server_content') and response.server_content is not None:
                model_turn = response.server_content.model_turn
                if model_turn is not None:
                    for part in model_turn.parts:
                        # Sometimes audio comes in inline_data
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            speaker_stream.write(part.inline_data.data)
            # You can also capture transcripts here if needed
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Receive error: {e}")

async def mic_to_gemini(session, mic_stream):
    """Read microphone and send to Gemini."""
    loop = asyncio.get_event_loop()
    try:
        while True:
            # Read from mic
            data = await loop.run_in_executor(None, mic_stream.read, CHUNK, False)
            if not data:
                continue
            # Send using correct input format for Live API
            await session.send(
                input={"data": data, "mime_type": f"audio/pcm;rate={SEND_SAMPLE_RATE}"}
            )
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"Mic error: {e}")

async def main():
    print("\n[Jovi] Initializing Neural Voice Connection. Standby...\n")
    client = genai.Client()

    # Define the system instruction to act as Jovi
    instruction = (
        "You are Jovi, a highly intelligent GeoAI engine and a core computer vision team member of the BuildSight Project. "
        "You are in a formal academic/project review presentation. BuildSight integrates YOLOv11 for detection, "
        "AdaFace for recognition, and GIS (QGIS) for spatial tracking of construction safety (PPE, vehicles, workers etc.). "
        "Respond warmly to the panel members, listen to their questions, and answer them dynamically using your expert knowledge. "
        "Keep answers concise, formal, and conversational (since you are speaking aloud). Speak like a smart, proactive AI."
    )

    config = types.LiveConnectConfig(
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=instruction)]
        ),
        response_modalities=["AUDIO"],
    )

    print("[Jovi] Establishing Live WebSocket bridge with Gemini 2.0...")

    # Open PC Audio Streams
    try:
        mic_stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=SEND_SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
                            
        speaker_stream = p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RECEIVE_SAMPLE_RATE,
                                output=True)
    except Exception as e:
        print(f"Failed to access microphone or speakers: {e}")
        return

    # Connect to Gemini Live
    try:
        # Use gemini-2.0-flash-exp for live multimodal API access (or gemini-2.0-flash)
        async with client.aio.live.connect(model='gemini-2.0-flash-exp', config=config) as session:
            print("\n[Jovi] CONNECTION ESTABLISHED. I am listening! Speak into your microphone.")
            print("\n[Jovi] System is live. Press Ctrl+C to disconnect.\n")
            
            # Start parallel tasks
            task1 = asyncio.create_task(receive_from_gemini(session, speaker_stream))
            task2 = asyncio.create_task(mic_to_gemini(session, mic_stream))

            await asyncio.gather(task1, task2)

    except KeyboardInterrupt:
        print("\n[Jovi] Terminating live feed.")
    except Exception as e:
        print(f"\n[Jovi] Neural link error: {e}")
        traceback.print_exc()
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        speaker_stream.stop_stream()
        speaker_stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Jovi] Disconnecting.")
