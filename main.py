from Parser import Parser
from Processor import Processor
from Speech import TTS, STT

# p = Parser("/home/daniel/Downloads/chatbot_nlp/data")
p = Parser("/home/daniel/Downloads/simplified-nq-train.jsonl")
q, a = p.parse_google()
p.store_data(q, a)

bot = Processor()
bot.main(q, a)
bot.save_model(bot.encoder, name="google_enc.h5")
bot.save_model(bot.decoder, name="google_dec.h5")
bot.save_tokenizer(bot.tokenizer, name="google.token.pickle")
# bot.load_all("encoder.h5", "decoder.h5", "tokenizer.pickle")

tts = TTS()
stt = STT()

while True:
    inp = stt.speech_to_text()
    ans = bot.ask_question(inp)
    print("The bot said: " + ans)
    tts.text_to_speech(ans)
