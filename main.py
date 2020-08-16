from Parser import Parser
from Processor import Processor

# p = Parser("/home/daniel/Downloads/train-v2.0.json")
parser = Parser("/home/daniel/Downloads/chatbot_nlp/data")
q, a = parser.main()
bot = Processor()
bot.main(q, a)
# encoder = bot.load_model("encoder.h5")
# decoder = bot.load_model("decoder.h5")
# tokenizer = bot.load_tokenizer("tokenizer.pickle")
# bot.converse(encoder, decoder, tokenizer)
