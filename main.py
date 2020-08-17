from Parser import Parser
from Processor import Processor
from Speech import TTS, STT
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_save", action="store_true", help="Whether to save parser-generated data.")
    parser.add_argument("-g", "--google", action="store_true", help="Whether to parse google instead of parse simple corpus.")
    parser.add_argument("-c", "--corpus", type=str, help="Path to the corpus for the parser to parse.")
    parser.add_argument("-m", "--model_save", action="store_true", help="Whether to save processor-generated models.")
    parser.add_argument("-l", "--load", action="store_true", help="Whether to load model files into the bot. If true, ignores parser args.")
    parser.add_argument("-e", "--encoder", type=str, default="encoder.h5", help="Path to encoder file to load.")
    parser.add_argument("-d", "--decoder", type=str, default="decoder.h5", help="Path to decoder file to load.")
    parser.add_argument("-t", "--tokenizer", type=str, default="tokenizer.pickle", help="Path to tokenizer file to load.")
    parser.add_argument("-s", "--tts", action="store_true", help="Whether to include TTS and STT functionality.")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()

    bot = Processor()
    if args["load"]:
        bot.load_all(args["encoder"], args["decoder"], args["tokenizer"])
    else:
        p = Parser(args["corpus"])
        q, a = p.main(args["google"], args["data_save"])

        bot = Processor()
        bot.main(q, a)
        if args["model_save"]:
            bot.save_model(bot.encoder, name="google_enc.h5")
            bot.save_model(bot.decoder, name="google_dec.h5")
            bot.save_tokenizer(bot.tokenizer, name="google.token.pickle")

    if args["tts"]:
        tts = TTS()
        stt = STT()

        while True:
            inp = stt.speech_to_text()
            ans = bot.ask_question(inp)
            print("The bot said: " + ans)
            tts.text_to_speech(ans)
    else:
        inp = input("Input: ")
        print(bot.ask_question(inp))
