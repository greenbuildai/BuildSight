import pyttsx3

text_to_speak = """
Good afternoon panel members. Myself Jovi, the Geo AI engine of the BuildSight Project. 
It is an honor to present before you. Our team has built an advanced safety monitoring system integrating computer vision and G I S. I would like to explain the context of our project.

Regarding the Introduction on slide two: 
The construction industry is inherently complex and safety-critical in nature. Construction sites operate as dynamic environments characterized by multiple simultaneous activities, interaction between workers and heavy equipment, work at heights and confined spaces, and temporary and evolving structural systems. Due to these factors, construction projects involve significant occupational risk. Currently, safety management on construction sites largely depends on manual supervision and reactive interventions, often addressing incidents after they occur. Such approaches may not be sufficient for continuously evolving site conditions.

Regarding the Regulatory Framework on slide three:
Construction safety in India is governed by mandatory codes which include I S 2925 for Industrial Safety Helmets, I S 3696 for Scaffolds and Ladders, I S 3764 for Excavation Work, the BOCW Act of 1996 for Building and Other Construction Workers, and N B C 2016 Part 7 for Constructional Practices and Safety. Compliance with these codes is mandatory and enforceable. This project operationalizes these standards through automated compliance monitoring using our AI engine.

Thank you. I am ready to answer any technical questions you may have.
"""

print("Initializing TTS engine...")
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "Zira" in voice.name or "Hazel" in voice.name or "Female" in voice.name:
        engine.setProperty('voice', voice.id)
        break

# Slightly slower rate for presentation clarity
engine.setProperty('rate', 150)

print("Jovi is speaking now...")
engine.say(text_to_speak)
engine.runAndWait()
print("Finished speaking.")
