import streamlit as st
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

st.title('Language Translation App')
st.write('Translate English to Hindi')

# User input
input_text = st.text_area('Enter text in English:', '')

max_len_eng = 56
max_len_hin = 62

with open('app/word2idx_outputs.pkl', 'rb') as file:
    word2idx_outputs = pickle.load(file)

with open('app/idx2word_outputs.pkl', 'rb') as file:
    idx2word_outputs = pickle.load(file)

def decontractions(phrase):

    # decontractions for english language
    contractions = {
            "won't": "will not",
            "can't": "can not",
            "n't": " not",
            "'re": " are",
            "'s": " is",
            "'d": " would",
            "'ll": " will",
            "'t": " not",
            "'ve": " have",
            "'m": " am"
                    }
    for contraction, replacement in contractions.items():
        phrase = re.sub(contraction, replacement, phrase)
    return phrase

def preprocess(text):

    text = text.lower() # converts the text to lower case.
    text = decontractions(text) # applies the function decontractions on the text.
    text = re.sub('[^A-Za-z]+', ' ', text) # removes all non-alphabetic characters
                                              # from the text string, leaving only letters
                                              # (either uppercase or lowercase).
    text = ' '.join([word for word in text.split() if len(word) > 1]) # Removing single-letter words.
    stop_words = set(stopwords.words('english')) # extracting stopwords from english language
    text = ' '.join([word for word in text.split() if word not in stop_words]) # removes stopwords from
                                                                               # the text string and joins
                                                                               # the remaining words into a
                                                                               # single string separated by spaces.

    return text


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Assuming you have defined vocab_size_eng, vocab_size_hin, train_dataloader, etc.

# Define input layers
input_eng = Input(shape=(56,), name='input_eng')
input_hin = Input(shape=(62,), name='input_hin')

# Instantiate the model
model = Encoder_decoder(vocab_size_eng=8471,
                        vocab_size_hin=9495,
                        embedding_dim_eng=150,
                        embedding_dim_hin=150,
                        input_length_eng=56,
                        input_length_hin=62,
                        encoder_units=32,
                        decoder_units=32)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def translate(input_text):
    input_sequence    = preprocess(input_text)
    with open('app/tokenizer_eng.pickle', 'rb') as file:
        tokenize_eng = pickle.load(file)
    input_sequence    = pad_sequences(tokenize_eng.texts_to_sequences([input_sequence]), maxlen=max_len_eng, dtype='int32', padding='post')
    model.load_weights('app/custom_model.h5')
    en_h,en_c         = model.layers[0].initialize_states(1)
    en_outputs        = model.layers[0](tf.constant(input_sequence), [en_h,en_c])
    de_input          = tf.constant([[word2idx_outputs['<start>']]])
    de_state_h, de_state_c = en_outputs[1:]
    out_words = []
    while True:
        de_output, de_state_h, de_state_c = model.layers[1](de_input, [de_state_h, de_state_c])
        output      =   model.layers[2](de_output)
        output      =   tf.argmax(output, -1)
        out_words.append(idx2word_outputs[output.numpy()[0][0]+1])
        if out_words[-1] == '<end>' or len(out_words) >= max_len_hin:
          break
        de_input    = output
    return ' '.join(out_words)

# Button to perform translation
if st.button('Translate'):
    if input_text:
        translated_text = translate(input_text)
        st.write('### Translated Text:')
        st.write(translated_text)
    else:
        st.write('Please enter some text to translate.')
