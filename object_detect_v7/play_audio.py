import pyttsx3

# Text you want to convert to speech
text = "Hello, how are you today?"

# Initialize the TTS engine
engine = pyttsx3.init()

# Set properties (optional)
engine.setProperty('rate', 150)  # Speed of speech

# Convert text to speech
engine.say(text)

# Wait for the speech to finish
engine.runAndWait()
