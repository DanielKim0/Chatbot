import speech_recognition as sr
import pyttsx3


class TTS:
    def __init__(self):
        self.tts = pyttsx3.init()

    def text_to_speech(self, text):
        self.tts.say(text)
        self.tts.runAndWait()


class STT:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()

    def speech_to_text(self):
        with self.mic as source:
            audio = self.rec.listen(source)
        return self.rec.recognize_google(audio)
