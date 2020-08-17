import os
from chatterbot import ChatBot
from chatterbot.trainers import UbuntuCorpusTrainer, ChatterBotCorpusTrainer


class ChatterbotBot:
    """This is a (very simple) conversational bot using chatterbot, a popular open-source chatbot framework. The bot
    can be trained easily. This is just a 'tech-demo' that could be fleshed out to a higher degree, such as adding more
    options, flexibility, and documentation. However, as this was a quick and easy option to build up, I decided to give
    it its own file for testing and display purposes.."""

    def __init__(self):
        self.bot = ChatBot('ChatterbotBot',
            storage_adapter='chatterbot.storage.SQLStorageAdapter',
            preprocessors=['chatterbot.preprocessors.clean_whitespace'],
            logic_adapters=['chatterbot.logic.MathematicalEvaluation', 'chatterbot.logic.TimeLogicAdapter'],
            database_uri='sqlite:///chatterbotbot.db'
        )

        if not os.path.exists("chatterbotbot.db"):
            self.train()

    def train(self, path=None):
        if not path:
            trainer = UbuntuCorpusTrainer(self.bot)
            trainer.train()
        else:
            trainer = ChatterBotCorpusTrainer(self.bot)
            trainer.train(path)

    def talk(self, statement):
        return self.bot.get_response(statement)

    def main(self):
        while True:
            statement = input()
            print("\n" + self.talk(statement) + "\n")


if __name__ == "__main__":
    bot = ChatterbotBot()
    bot.main()