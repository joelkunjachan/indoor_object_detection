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

def predict():
    text = str()
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Adjust settings for better accuracy
    recognizer.energy_threshold = 300  # Lower threshold for quieter environments
    recognizer.dynamic_energy_threshold = True  # Adjust dynamically
    recognizer.pause_threshold = 0.8  # Shorter pause to detect end of speech
    recognizer.phrase_threshold = 0.3  # Sensitivity for phrase detection
    recognizer.non_speaking_duration = 0.5  # Duration of non-speaking audio to consider as silence

    # Define the input type (microphone)
    input_type = 'microphone'

    if input_type == 'microphone':
        # Use microphone input
        with sr.Microphone() as source:
            print("Speak something...")
            text_to_speech("Say the object name")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Listen for speech with timeout
            try:
                audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                print("Timeout: No speech detected.")
                text_to_speech("Timeout: No speech detected.")
                return ""

    # Perform speech recognition
    try:
        print("Recognizing...")
        text_to_speech("Recognizing audio")
        # Recognize the speech using Google Speech Recognition with language specification
        text = recognizer.recognize_google(audio_data, language='en-IN') # Changed from 'en-US'
        print("You said:", text)
        text_to_speech("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand audio.")
        text_to_speech("Sorry, I could not understand audio.")
        # Retry once
        # print("Retrying...")
        # text_to_speech("Retrying")
        # try:
        #     with sr.Microphone() as source:
        #         recognizer.adjust_for_ambient_noise(source, duration=1)
        #         audio_data = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        #     text = recognizer.recognize_google(audio_data, language='en-IN')
        #     print("You said:", text)
        #     text_to_speech("You said: " + text)
        # except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
        #     text = ""
    except sr.RequestError as e:
        print("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
        text_to_speech("Error: Could not request results from Google Speech Recognition service; {0}".format(e))
        text = ""

    return text

    return text
