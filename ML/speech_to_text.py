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

    # List of valid object keywords
    object_list = ["person", "bench", "backpack", "umbrella", "handbag", "bottle",
                   "glass", "cup", "knife", "spoon", "bowl", "chair", "bed", "toilet",
                   "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "book"]

    # Load the Vosk model
    #model_path = os.path.join(os.path.dirname(__file__), 'vosk-model')
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
        # Check for keywords in the recognized text
        words = text.lower().split()
        for word in words:
            if word in [obj.lower() for obj in object_list]:
                print("Detected object:", word)
                text_to_speech("Detected object: " + word)
                return word
        # If no keyword found
        print("No valid object detected in speech.")
        text_to_speech("No valid object detected in speech.")
        return ""
    else:
        print("Could not understand audio.")
        text_to_speech("Sorry, I could not understand audio.")
        return ""

