import speech_recognition as sr
import pyttsx3
import os


class Speech:
    def __init__(self):
        self.tts = pyttsx3.init()
        pass

    def speech_to_text(self):
        pass

    def text_to_speech(self, text):
        self.tts.say(text)
        self.tts.runAndWait()

    def main(self):
        self.text_to_speech("Hello world!")


s = Speech()
s.main()
