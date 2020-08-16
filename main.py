from Parser import Parser
from Processor import Processor

# p = Parser("/home/daniel/Downloads/train-v2.0.json")
parser = Parser("/home/daniel/Downloads/chatbot_nlp/data")
q, a = parser.main()
bot = Processor()
# bot.main(q, a)
bot.converse_preload("encoder.h5", "decoder.h5", "tokenizer.pickle")
