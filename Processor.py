import numpy as np
import tensorflow as tf
import keras
from keras import preprocessing, utils
import pickle

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class Processor:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.tokenizer = None

    def clean_data(self, questions, answers):
        """Function that cleans and tags question/answer data for use later."""
        answers = ["<START> " + answer + " <END>" for answer in answers]
        return questions, answers

    def create_tokenizer(self, questions, answers):
        """Function that creates a tokenizer using the question/answer data. Tokenizer vectorizes a set of words into
        a numerical vector, which then can be read into the machine-learning model."""
        tokenizer = preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(questions + answers)
        vocab_size = len(tokenizer.word_index) + 1
        vocab = [word for word in tokenizer.word_index]
        return tokenizer, vocab, vocab_size

    def prep_data(self, tokenizer, questions, answers, vocab_size):
        """Function that takes the question/answer data and preps it, which includes tokenizing and padding the data
        to make it readable and separating it for use by the decoder and encoder."""
        tokenized_questions = tokenizer.texts_to_sequences(questions)
        maxlen_questions = max([len(x) for x in tokenized_questions])
        padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding="post")
        encoder_input_data = np.array(padded_questions)

        tokenized_answers = tokenizer.texts_to_sequences(answers)
        maxlen_answers = max([len(x) for x in tokenized_answers])
        padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding="post")
        decoder_input_data = np.array(padded_answers)

        tokenized_answers = tokenizer.texts_to_sequences(answers)
        for i in range(len(tokenized_answers)):
            tokenized_answers[i] = tokenized_answers[i][1:]
        padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding="post")
        onehot_answers = utils.to_categorical(padded_answers, vocab_size)
        decoder_output_data = np.array(onehot_answers)

        return (encoder_input_data, decoder_input_data, decoder_output_data), maxlen_questions, maxlen_answers

    def create_encoder(self, maxlen_questions, vocab_size):
        """Function that creates encoder data according to a seq-to-seq model. Encoders process the input and return state
        vectors that are used later on by the decoder."""
        encoder_inputs = tf.keras.layers.Input(shape=(maxlen_questions, ))
        encoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, mask_zero=True)(encoder_inputs)
        encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(encoder_embedding)
        encoder_states = [state_h, state_c]
        return encoder_inputs, encoder_states

    def create_decoder(self, maxlen_answers, vocab_size, encoder_states):
        """Function that creates decoder data according to a seq-to-seq model. Decoders are trained to predict the next
        values of a target sequence, in this case the inputted encoder's data."""
        decoder_inputs = tf.keras.layers.Input(shape=(maxlen_answers,))
        decoder_embedding = tf.keras.layers.Embedding(vocab_size, 200, mask_zero=True)(decoder_inputs)
        decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)
        output = decoder_dense(decoder_outputs)
        return decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense, output

    def create_model(self, model_data, encoder_inputs, decoder_inputs, output, batch_size=128, epochs=256):
        """Function that actually creates and trains the seq-to-seq model using the encoder/decoder data provided."""
        encoder_input_data, decoder_input_data, decoder_output_data = model_data
        model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="categorical_crossentropy")
        model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=batch_size, epochs=epochs)
        return model

    def encoder_inference(self, encoder_inputs, encoder_states):
        """Function that creates a front-facing encoder inference model that converts inputted data to states."""
        return tf.keras.models.Model(encoder_inputs, encoder_states)

    def decoder_inference(self, decoder_inputs, decoder_embedding, decoder_lstm, decoder_dense):
        """Function that creates a front-facing decoder inference model that pairs with the encoder model."""
        decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(200,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_states = [state_h, state_c]
        return tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def tokenize(self, sentence, tokenizer):
        """Function that tokenizes an inputted sentence using the tokenizer generated earlier."""
        words = sentence.lower().split()
        tokens = []
        for word in words:
            if word in tokenizer.word_index:
                tokens.append(tokenizer.word_index[word])
            else:
                return None
        return preprocessing.sequence.pad_sequences([tokens], padding="post")

    def ask_question(self, inp):
        """Function that takes in an input and generates an output using the seq-to-seq built and trained earlier. After
        being given an input, function tokenizes input and passes it through the encoder and decoder, formatting the
        values in-between to make it readable by the models."""
        if not inp:
            print("Input empty!")
            return None

        tokenized = self.tokenize(inp, self.tokenizer)
        if tokenized is None:
            print("Sorry! The bot could not understand your input.")
            return None
        state_values = self.encoder.predict(tokenized)
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = self.tokenizer.word_index["start"]

        stop = False
        decoded = ""
        while not stop:
            dec_outputs, h, c = self.decoder.predict([empty_target_seq] + state_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None

            for word, index in self.tokenizer.word_index.items():
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
        return keras.models.load_model(name, compile=False)

    def save_tokenizer(self, model, name="tokenizer.pickle"):
        with open(name, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_tokenizer(self, name="tokenizer.pickle"):
        with open(name, "rb") as handle:
            return pickle.load(handle)

    def load_all(self, encoder, decoder, tokenizer):
        self.encoder = self.load_model(encoder)
        self.decoder = self.load_model(decoder)
        self.tokenizer = self.load_tokenizer(tokenizer)

    def chatbot_prep(self, questions, answers):
        """Function that prepares the chatbot for talking, building models and training them on the inputted question
        and answer data. Data should be inputted in two lists."""
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
        self.encoder, self.decoder, self.tokenizer = self.chatbot_prep(questions, answers)
        while True:
            inp = input("Enter text: ")
            print(self.ask_question(inp))
