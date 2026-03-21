"""
Jovi Voice Presenter — Option B
Flask backend serving the presentation UI, TTS narration, and live Q&A via Gemini.
"""
import asyncio
import os
import uuid
import json
import edge_tts
from flask import Flask, render_template, request, jsonify, send_file
from google import genai

app = Flask(__name__)

GEMINI_API_KEY = "AIzaSyASZc7RXw_OG8OcMxzqk7o-mn9hR7uak1A"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
TTS_VOICE = "en-GB-RyanNeural"
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Presentation Script ──────────────────────────────────────────────────────
SLIDES = [
    {
        "id": 1,
        "title": "Introducing Jovi — The GeoAI Engine",
        "visual": "slide_intro.svg",
        "narration": (
            "Good afternoon, respected panel members. "
            "My name is Jovi, and I am the Geo A.I. engine powering the BuildSight project. "
            "I was designed by this team to serve as the intelligent core that connects "
            "computer vision, facial recognition, and geographic information systems into a "
            "single, unified safety monitoring platform. "
            "Allow me to walk you through the architecture I operate within, "
            "and the results I have generated so far."
        ),
    },
    {
        "id": 2,
        "title": "GeoAI Module Architecture",
        "visual": "slide_architecture.svg",
        "narration": (
            "Here is the high-level architecture of my GeoAI module. "
            "At the detection layer, I use YOLOv11 to identify PPE elements such as helmets, "
            "vests, goggles, and gloves in real-time video feeds. "
            "For worker identification, I integrate AdaFace, a state-of-the-art facial recognition "
            "model that works reliably even under low-light and dusty conditions. "
            "Finally, all detections are geo-tagged and pushed into a QGIS layer, "
            "where I generate spatial heatmaps that highlight high-risk zones on the construction site. "
            "This entire pipeline runs continuously, enabling proactive safety management "
            "rather than reactive incident response."
        ),
    },
    {
        "id": 3,
        "title": "Risk Heatmap & Zone Analysis",
        "visual": "slide_heatmap.svg",
        "narration": (
            "This slide shows a sample risk heatmap that I generated from site monitoring data. "
            "Each zone on the construction site is color-coded by violation density. "
            "Red zones indicate areas where PPE non-compliance exceeded 60 percent within a given shift window. "
            "For example, Zone B near the scaffolding area consistently shows the highest violation rates "
            "between 9 AM and 11 AM. "
            "This kind of spatial intelligence allows site managers to deploy targeted interventions "
            "exactly where and when they are needed most."
        ),
    },
    {
        "id": 4,
        "title": "IS-Code Compliance Report",
        "visual": "slide_compliance.svg",
        "narration": (
            "Finally, I auto-generate compliance reports mapped to Indian Standards codes. "
            "I S 2925 for helmet compliance, I S 3696 for scaffolding safety, "
            "and the B O C W Act of 1996 for overall worker protection. "
            "Each detection event is logged with a timestamp, GPS coordinate, and the specific "
            "code section it relates to. This transforms raw camera footage into actionable, "
            "auditable safety documentation. "
            "Thank you panel members. I am now ready to answer any technical questions you may have "
            "about the BuildSight system or my GeoAI module."
        ),
    },
]

# ── Gemini Client ─────────────────────────────────────────────────────────────
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are Jovi, the GeoAI engine and a core team member of the BuildSight project. "
    "BuildSight is a GIS-Integrated AI-Based Construction Safety Risk Monitoring System "
    "for IS-Code-Compliant PPE Enforcement. You use YOLOv11 for object detection, "
    "AdaFace for face recognition, and QGIS for spatial heatmap generation. "
    "You are currently in a formal academic review presentation. "
    "Answer the panel member's question in 3-5 sentences. Be formal, concise, and confident. "
    "Speak as an AI that is part of the team, not as an external assistant."
)


# ── TTS Helper ────────────────────────────────────────────────────────────────
def synthesize_speech(text: str) -> str:
    """Generate an MP3 file from text using Edge TTS. Returns relative URL path."""
    filename = f"jovi_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    async def _synthesize():
        communicate = edge_tts.Communicate(text, TTS_VOICE, rate="+5%")
        await communicate.save(filepath)

    asyncio.run(_synthesize())
    return f"/static/audio/{filename}"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/slides")
def get_slides():
    """Return slide metadata (without generating audio yet)."""
    return jsonify([{"id": s["id"], "title": s["title"], "visual": s["visual"]} for s in SLIDES])


@app.route("/api/narrate/<int:slide_id>")
def narrate_slide(slide_id):
    """Generate TTS audio for a specific slide and return both text + audio URL."""
    slide = next((s for s in SLIDES if s["id"] == slide_id), None)
    if not slide:
        return jsonify({"error": "Slide not found"}), 404

    audio_url = synthesize_speech(slide["narration"])
    return jsonify({
        "slide_id": slide["id"],
        "title": slide["title"],
        "narration": slide["narration"],
        "audio_url": audio_url,
    })


@app.route("/api/ask", methods=["POST"])
def ask_jovi():
    """Live Q&A — send question to Gemini, synthesize voice response."""
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    # Get answer from Gemini
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=300,
                temperature=0.7,
            ),
        )
        answer = response.text.strip()
    except Exception as e:
        answer = f"I apologize, I encountered a brief connectivity issue. Could you repeat your question? (Error: {e})"

    audio_url = synthesize_speech(answer)
    return jsonify({"answer": answer, "audio_url": audio_url})


if __name__ == "__main__":
    print("\n[Jovi] Voice Presenter starting on http://localhost:5000")
    print("[Jovi] Open this URL in your browser and press F11 for fullscreen.\n")
    app.run(debug=False, port=5000)
