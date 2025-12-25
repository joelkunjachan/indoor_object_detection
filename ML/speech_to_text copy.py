import pyttsx3
import pyaudio
import json
from vosk import Model, KaldiRecognizer


def text_to_speech(text):
    """Convert text to speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speech speed
    engine.say(text)
    engine.runAndWait()


def predict():
    """Recognize speech using Vosk."""
    text = ""

    # Load the Vosk model
    model = Model("vosk-model-small-en-us-0.15")
    recognizer = KaldiRecognizer(model, 16000)

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    print("Speak something...")
    text_to_speech("Say object name")

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
               break  # Stop after receiving one full phrase

    stream.stop_stream()
    stream.close()
    p.terminate()

    if text:
        print("You said:", text)
        text_to_speech("You said: " + text)
    else:
        print("Could not understand audio.")
        text_to_speech("Sorry, I could not understand audio.")

    return text

