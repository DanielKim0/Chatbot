from Parser import Parser
from Processor import Processor

# p = Parser("/home/daniel/Downloads/train-v2.0.json")
parser = Parser("/home/daniel/Downloads/chatbot_nlp/data")
q, a = parser.main()
bot = Processor(q, a)
bot.main()
