# Chatbot
A chatbot written in Python powered by an encoder-decoder/seq2seq model. Inclues functionality for parsing corpus data, training the model on inputted question/answer data, and text-to-speech and speech-to-text conversions for actually speaking with the bot.

# Requirements
python 3.7+ with modules tensorflow, keras, speech_recognition, pyttsx3, yaml
TTS dependent on installing sapi5 (Windows), nsss (MacOS X), or espeak (Linux).
STT dependent on installing pyaudio (all OSes).

# Usage
Call ```python main.py``` with parameters, listed in the table below.

Parameter | Default | Description
-s, --tts | False | Whether to include TTS and STT functionality.
-d, --data_save | False | Whether to save parser-generated data.
-g, --google | False | Whether to parse google corpus instead of parse simple corpus.
-c, --corpus | N/A | Path to the corpus for the parser to parse.
-m, --model_save | False | Whether to save processor-generated models.
-l, --load | False | Whether to load model files into the bot. If true, ignores parser args. If false, ignores below args.
-e, --encoder | "encoder.h5" | Path to encoder file to load.
-d, --decoder | "decoder.h5" | Path to decoder file to load.
-t, --tokenizer | "tokenizer.pickle" | Path to tokenizer file to load.

Note: "corpus" argument mandatory if "load" is False.