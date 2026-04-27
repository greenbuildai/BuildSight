import subprocess
import os

text_to_speak = """Good afternoon, respected panel members. My name is Jovi, and I am the Geo A.I. engine and a core team member of the BuildSight Project. It is an absolute honor to present before you today.

If I may draw your attention to our first few slides, I'd like to set the context for our work. 

As you can see on slide two, the construction industry is inherently complex and safety-critical. Construction sites are incredibly dynamic environments. At any given moment, there are multiple simultaneous activities, dangerous interactions between workers and heavy equipment, critical work happening at heights, and temporary structural systems that are constantly evolving. All of these factors compound to create a massive amount of occupational risk. 

Currently, safety management relies heavily on manual supervision and reactive interventions... addressing incidents only after they occur. But because site conditions evolve so quickly, these manual approaches just aren't sufficient anymore.

Moving to slide three, we look at the regulatory framework in India. Construction safety is governed by strict, mandatory codes like I S 2925 for safety helmets, and the N B C 2016 Part 7 for constructional practices. Compliance isn't optional—it's legally enforceable. 

Our project, BuildSight, exists to operationalize these exact standards through continuous, automated compliance monitoring using our computer vision and G I S architecture.

Thank you. I am now ready to take any technical questions you may have."""

print("\n[Jovi] Synthesizing human voice via Neural Engine...\n")

# Using Microsoft Cognitive Edge TTS via CLI for a very human UK voice (like Daniel)
# en-GB-RyanNeural is a British male voice that is incredibly professional.
with open("presentation_text.txt", "w", encoding="utf-8") as f:
    f.write(text_to_speak)
    
subprocess.run([
    "edge-tts", 
    "--voice", "en-GB-RyanNeural", 
    "--rate", "+5%", 
    "-f", "presentation_text.txt", 
    "--write-media", "presentation_daniel.mp3"
])

print("[Jovi] Voice synthesized securely.")
print("[Jovi] Playing human presentation from laptop speakers...\n")

# Async playback on Windows using default media player
os.system("start presentation_daniel.mp3")
