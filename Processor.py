import numpy as np
import tensorflow as tf
import keras
from keras import preprocessing, utils
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class Processor:
    def __init__(self):
        pass

    def clean_data(self, questions, answers):
        answers = ["<START> " + answer + " <END>" for answer in answers]
        return questions, answers

    def create_tokenizer(self, questions, answers):
        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(questions + answers)
        vocab_size = len(tokenizer.word_index) + 1
        vocab = [word for word in tokenizer.word_index]
        return tokenizer, vocab, vocab_size

    def prep_data(self, tokenizer, questions, answers, vocab_size):
        tokenized_questions = tokenizer.texts_to_sequences(questions)
        maxlen_questions = max([len(x) for x in tokenized_questions])
        padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding="post")
        encoder_input_data = np.array(padded_questions)

        tokenized_answers = tokenizer.texts_to_sequences(answers)        tokens = [tokenizer.word_index[word] for word in words]
        tokenized_answers = tokenizer.texts_to_sequences(answers)
        for i in range(len(tokenized_answers)):
            tokenized_answers[i] = tokenized_answers[i][1:]
        padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding="post")
        onehot_answers = utils.to_categorical(padded_answers, vocab_size)
        decoder_output_data = np.array(onehot_answers)

        return (encoder_input_data, decoder_input_data, decoder_output_data), maxlen_questions, maxlen_answers

    def create_encoder(self, maxlen_questions, vocab_size):
        encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions, ))
        encoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, mask_zero=True)(encoder_inputs)
        encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(encoder_embedding)
        encoder_states = [state_h, state_c]
        return encoder_inputs, encoder_states

    def create_decoder(self, maxlen_answers, vocab_size, encoder_states):
        decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers,))
        decoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, mask_zero=True)(decoder_inputs)
        decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)
        output = decoder_dense(decoder_outputs)
        return decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense, output

    def create_model(self, model_data, encoder_inputs, decoder_inputs, output, batch_size=50, epochs=150):
        encoder_input_data, decoder_input_data, decoder_output_data = model_data
        model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="categorical_crossentropy")
        model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=batch_size, epochs=epochs)
        return model

    def encoder_inference(self, encoder_inputs, encoder_states):
        return tf.keras.models.Model(encoder_inputs, encoder_states)

    def decoder_inference(self, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense):
        decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(200,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_states = [state_h, state_c]
        return tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def tokenize(self, sentence, tokenizer):
        words = sentence.lower().split()
        tokens = []
        for word in words:
            if word in tokenizer.word_index:
                tokens.append(tokenizer.word_index[word])
            else:
                return None
        return preprocessing.sequence.pad_sequences([tokens], padding="post")

    def converse_preload(self, encoder_file, decoder_file, tokenizer_file):
        encoder = self.load_model(encoder_file)
        decoder = self.load_model(decoder_file)
        tokenizer = self.load_tokenizer(tokenizer_file)
        bot.converse(encoder, decoder, tokenizer)

    def converse(self, encoder, decoder, tokenizer):
        while True:
            answer = self.ask_question(encoder, decoder, tokenizer)
            if answer:
                print(answer)

    def ask_question(self, encoder, decoder, tokenizer):
        inp = input("Your input: ")
        if not inp:
            print("Input empty!")
            return None

        tokenized = self.tokenize(inp, tokenizer)
        if not tokenized:
            print("Sorry! The bot could not understand your input.")
            return None
        state_values = encoder.predict(tokenized)
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index["start"]

        stop = False
        decoded = ""
        while not stop:
            dec_outputs, h, c = decoder.predict([empty_target_seq] + state_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None

            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded += " {}".format(word)
                    sampled_word = word

            if sampled_word == "end":
                stop = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            state_values = [h, c]

        # remove the "end" tag
        return decoded[:-4]

    def save_model(self, model, name="model.h5"):
        model.save(name)

    def load_model(self, name="model.h5"):
        return keras.models.load_model(name)

    def save_tokenizer(self, model, name="tokenizer.pickle"):
        with open(name, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self, name="tokenizer.pickle"):
        with open(name, "rb") as handle:
            return pickle.load(handle)

    def chatbot_prep(self, questions, answers):
        questions, answers = self.clean_data(questions, answers)
        tokenizer, vocab, vocab_size = self.create_tokenizer(questions, answers)
        model_data, maxlen_questions, maxlen_answers = self.prep_data(tokenizer, questions, answers, vocab_size)
        encoder_inputs, encoder_states = self.create_encoder(maxlen_questions, vocab_size)
        decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense, output = self.create_decoder(maxlen_answers, vocab_size, encoder_states)
        self.create_model(model_data, encoder_inputs, decoder_inputs, output)
        encoder = self.encoder_inference(encoder_inputs, encoder_states)
        decoder = self.decoder_inference(decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense)
        return encoder, decoder, tokenizer

    def main(self, questions, answers):
        encoder, decoder, tokenizer = self.chatbot_prep(questions, answers)
        self.save_model(encoder, "encoder.h5")
        self.save_model(decoder, "decoder.h5")
        self.save_tokenizer(tokenizer)
        self.converse(encoder, decoder, tokenizer)
