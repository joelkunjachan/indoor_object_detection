import speech_recognition as sr
import pyttsx3

def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()

    # Set properties (optional)
    engine.setProperty('rate', 150)  # Speed of speech

    # Convert text to speech
    engine.say(text)

    # Wait for the speech to finish
    engine.runAndWait() 

# Initialize the recognizer
recognizer = sr.Recognizer()

# Define the input type (microphone)
input_type = 'microphone'

if input_type == 'microphone':
    # Use microphone input
    with sr.Microphone() as source:
        print("Speak something...")
        text_to_speech("say object name")
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        # Listen for speech
        audio_data = recognizer.listen(source)

# Perform speech recognition
try:
    print("Recognizing...")
    text_to_speech("Recognizinge audio")
    # Recognize the speech using Google Speech Recognition
    text = recognizer.recognize_google(audio_data)
    print("You said:", text)
    text_to_speech("You said:" + text)
except sr.UnknownValueError:
    print("Sorry, I could not understand audio.")
    text_to_speech("Sorry, I could not understand audio.")
except sr.RequestError as e:
    print("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
    text_to_speech("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
