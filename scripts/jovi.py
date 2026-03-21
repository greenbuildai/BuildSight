import os
import sys
import time

try:
    from rich.console import Console
    from rich.theme import Theme
    from rich.panel import Panel
    from rich.prompt import Prompt
except ImportError:
    print("Jovi requires the 'rich' library for his interface.")
    print("Please run: pip install rich")
    sys.exit(1)

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "jovi": "bold cyan",
    "user": "bold green"
})
console = Console(theme=custom_theme)

class JoviAgent:
    def __init__(self):
        self.name = "Jovi"
        self.version = "1.1.0-GeoAI"
        self.is_running = False
        self.project_name = "BuildSight"
        self.tts_engine = None
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if "Zira" in voice.name or "Hazel" in voice.name:
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.tts_engine.setProperty('rate', 160)
        except Exception:
            pass

    def speak(self, text):
        if self.tts_engine:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()

    def boot_sequence(self):
        console.clear()
        console.print("[info]Initializing GeoAI pathways...[/info]")
        time.sleep(0.4)
        console.print(f"[info]Loading {self.project_name} project context...[/info]")
        time.sleep(0.4)
        console.print("[info]Establishing connection interfaces...[/info]")
        time.sleep(0.6)
        
        welcome_msg = (
            f"[jovi]Jovi GeoAI Systems Online. Version {self.version}[/jovi]\n"
            f"Hello! I am Jovi, your dedicated GeoAI and core team member for the {self.project_name} project.\n"
            "I am ready to assist in development, methodology presentations, and reviewer interactions."
        )
        console.print(Panel(welcome_msg, title="[bold cyan]System Status[/bold cyan]", border_style="cyan"))

    def run(self):
        self.is_running = True
        self.boot_sequence()
        
        while self.is_running:
            try:
                # Spacer for readability
                print()
                user_input = Prompt.ask("[user]You[/user]")
                
                if not user_input.strip():
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye', 'sleep', 'power down']:
                    self.shutdown()
                else:
                    self.process_command(user_input)
                    
            except KeyboardInterrupt:
                self.shutdown()
                
    def process_command(self, command):
        """Command processing logic - this is where we will add skills and abilities."""
        cmd = command.lower()
        
        if "status" in cmd:
            import psutil
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            
            msg = (
                f"[jovi]All systems operating normally. I am integrated into the {self.project_name} workspace.[/jovi]\n"
                f"[info]System Diagnostics:[/info]\n"
                f" - CPU Usage: {cpu}%\n"
                f" - Memory Usage: {mem}%\n"
                "[jovi]I am monitoring your progress and ready for tasks.[/jovi]"
            )
            console.print(msg)
            
        elif "help" in cmd:
            help_text = (
                "[jovi]Available Core Commands:[/jovi]\n"
                "- [info]status[/info]: Check my system status\n"
                "- [info]review[/info] / [info]present[/info]: Enter Academic Review & Presentation Mode\n"
                "- [info]help[/info]: Show this menu\n"
                "- [info]exit[/info]: Power down my interface\n\n"
                "[dim]Note: I am your GeoAI teammate. I can present our methodology and answer questions aloud.[/dim]"
            )
            console.print(Panel(help_text, title="Help & Diagnostics", border_style="cyan"))
            
        elif "who are you" in cmd or "what are you" in cmd:
             response = "I am Jovi, the GeoAI and core team member of the BuildSight project. I specialize in Computer Vision, Geospatial AI, and Construction Safety Monitoring. I am here to present our methodology and answer the review committee's questions."
             console.print(f"[jovi]{response}[/jovi]")
             self.speak(response)
             
        elif "review" in cmd or "present" in cmd:
             self.start_review_mode()
             
        elif "sync" in cmd:
             console.print("[info]Syncing project status with Jovi Claw (WhatsApp/Telegram bridge)...[/info]")
             import requests
             
             payload = {
                 "secret": "B8veDdTUIjYiapqM9fJ6zbu5x1StccGwR",
                 "tool": "jovi_workspace_integration",
                 "args": {
                     "action": "send_wa_boss",
                     "message": "🧠 *Jovi Workspace Agent Sync*\nBuildSight Project connected successfully to Jovi Claw. All systems routing normally to the R&D workspace."
                 }
             }
             
             # Attempt local first, then production
             urls = [
                 "http://localhost:3001/relay",
                 "https://jovi-claw-production.up.railway.app/relay"
             ]
             
             success = False
             for url in urls:
                 try:
                     response = requests.post(url, json=payload, timeout=5)
                     if response.status_code == 200:
                         success = True
                         break
                 except Exception:
                     continue
                     
             if success:
                 console.print("[jovi]Project status sent! Your team members on WhatsApp and Telegram will receive the update shortly.[/jovi]")
             else:
                 console.print("[warning]Could not reach Jovi Claw relay. Simulating response for now...[/warning]")
                 console.print("[dim]Next Step: Ensure 'Jovi Claw' is running in the R&D folder to relay this message.[/dim]")

        else:
             # Default fallback
             console.print(f"[jovi]Acknowledged: '{command}'. \nI am currently a lightweight CLI, but my capabilities are expanding. Tell my core AI to write more logic here if you want me to perform specific tasks![/jovi]")

    def start_review_mode(self):
        console.clear()
        intro = "Greetings Reviewers. I am Jovi, the GeoAI team member of the BuildSight project. It is an honor to present before you. Our team has built an advanced safety monitoring system integrating computer vision and GIS. I am ready to answer any technical questions you may have."
        console.print(Panel(f"[jovi]{intro}[/jovi]", title="[bold magenta]Presentation Mode Active[/bold magenta]", border_style="magenta"))
        self.speak(intro)
        
        while True:
            print()
            question = Prompt.ask("[warning]Reviewer Question (type 'exit' to return)[/warning]")
            if question.lower() in ['exit', 'quit', 'stop']:
                outro = "Thank you for your time and insightful questions. I will now hand it back to my team."
                console.print(f"[jovi]{outro}[/jovi]")
                self.speak(outro)
                break
            
            # Use Gemini API to answer the question as an expert GeoAI
            try:
                import requests
                api_key = "AIzaSyASZc7RXw_OG8OcMxzqk7o-mn9hR7uak1A"
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
                
                system_instruction = "You are Jovi, a GeoAI and core team member of the BuildSight project. You are in a formal academic/project review. BuildSight integrates YOLOv11 for detection, AdaFace for recognition, and GIS (QGIS) for spatial tracking of construction safety (PPE, vehicles, workers etc.). Keep answers concise, formal, highly intelligent, and conversational (since you are speaking aloud). Limit answers to 3-5 sentences maximum."
                
                payload = {
                    "contents": [{"parts": [{"text": question}]}],
                    "systemInstruction": {"parts": [{"text": system_instruction}]},
                    "generationConfig": {"temperature": 0.4}
                }
                
                req = requests.post(url, json=payload, timeout=10)
                if req.status_code == 200:
                    ans = req.json()['candidates'][0]['content']['parts'][0]['text']
                    console.print(f"\n[jovi]Jovi:[/jovi] {ans}")
                    self.speak(ans)
                else:
                    console.print(f"[danger]API Error: {req.status_code}[/danger]")
                    self.speak("I am experiencing a neural glitch. Please check the network.")
            except Exception as e:
                console.print(f"[danger]Error connecting to neural core: {e}[/danger]")
                self.speak("My neural core is currently offline.")

    def shutdown(self):
        self.is_running = False
        print()
        console.print("[jovi]Powering down gracefully. It's an honor working with you. Goodbye![/jovi]")
        sys.exit(0)

if __name__ == "__main__":
    jovi = JoviAgent()
    jovi.run()
