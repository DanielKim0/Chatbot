import speech_recognition as sr
import pyttsx3


class TTS:
    def __init__(self):
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 150)

    def text_to_speech(self, text):
        self.tts.say(text)
        self.tts.runAndWait()


class STT:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()

    def speech_to_text(self):
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source)
            print("Speak into your microphone: ")
            audio = self.rec.listen(source)
            print("Done!")
        text = self.rec.recognize_google(audio)
        print("Your speech was recognized as: " + text)
        return text
