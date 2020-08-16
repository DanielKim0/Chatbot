from Parser import Parser
from Processor import Processor
from Speech import TTS, STT

# p = Parser("/home/daniel/Downloads/train-v2.0.json")
# p = Parser("/home/daniel/Downloads/chatbot_nlp/data")
# q, a = p.main()

bot = Processor()
# bot.main(q, a)
bot.load_all("encoder.h5", "decoder.h5", "tokenizer.pickle")

tts = TTS()
stt = STT()

while True:
    inp = STT.speech_to_text()
    ans = bot.ask_question(inp)
    TTS.text_to_speech(ans)
