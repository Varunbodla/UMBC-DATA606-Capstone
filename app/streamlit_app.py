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
from tensorflow.keras.layers import Input,Embedding, LSTM, Dense, Flatten

class Encoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns encoder-outputs,encoder_final_state_h,encoder_final_state_c
    '''

    def __init__(self,vocab_size,embedding_dim,encoder_units,input_length):
      super().__init__()

      self.vocab_size    = vocab_size
      self.embedding_dim = embedding_dim
      self.input_length  = input_length
      self.encoder_units = encoder_units
      self.embedding     = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           name="Encoder_Embedding_Layer")
      self.lstm          = LSTM(self.encoder_units, return_state=True, return_sequences=True, name="Encoder_LSTM")

    def call(self,input_sequence,states):

      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to encoder_lstm
          returns -- encoder_output, last time step's hidden and cell state
      '''
      input_embed                            = self.embedding(input_sequence)
      lstm_output,lstm_state_h, lstm_state_c = self.lstm(input_embed,initial_state=states)
      return lstm_output, lstm_state_h, lstm_state_c

    def initialize_states(self,batch_size):

      '''
      Given a batch size it will return intial hidden state and intial cell state.
      If batch size is 32- Hidden state is zeros of size [32,lstm_units], cell state zeros is of size [32,lstm_units]
      '''
      lstm_state_h = tf.zeros([batch_size,self.encoder_units],dtype=tf.dtypes.float32)
      lstm_state_c = tf.zeros([batch_size,self.encoder_units],dtype=tf.dtypes.float32)
      return lstm_state_h,lstm_state_c
class Decoder(tf.keras.Model):
    '''
    Encoder model -- That takes a input sequence and returns output sequence
    '''
    def __init__(self,vocab_size, embedding_dim, decoder_units,input_length):
      super().__init__()
      self.vocab_size       = vocab_size
      self.embedding_dim    = embedding_dim
      self.input_length     = input_length
      self.decoder_units    = decoder_units
      self.embedding     = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, input_length=self.input_length,
                           name="embedding_layer_decoder")
      self.lstm          = LSTM(self.decoder_units, return_state=True, return_sequences=True, name="Decoder_LSTM")

    def call(self,target_sentence,states):

      '''
          This function takes a sequence input and the initial states of the encoder.
          Pass the input_sequence input to the Embedding layer, Pass the embedding layer ouput to decoder_lstm

          returns -- decoder_output,decoder_final_state_h,decoder_final_state_c
      '''
      target_embed                                                = self.embedding(target_sentence)
      decoder_output,decoder_final_state_h,decoder_final_state_c  = self.lstm(target_embed,states)
      return decoder_output,decoder_final_state_h,decoder_final_state_c

class Encoder_decoder(tf.keras.Model):

    def __init__(self,vocab_size_eng,vocab_size_hin, embedding_dim_eng,
                 embedding_dim_hin, input_length_eng,input_length_hin, encoder_units,decoder_units):
      super().__init__()
      self.vocab_size_eng       =  vocab_size_eng
      self.vocab_size_hin       =  vocab_size_hin
      self.embedding_dim_eng    =  embedding_dim_eng
      self.embedding_dim_hin    =  embedding_dim_hin
      self.input_length_eng     =  input_length_eng
      self.input_length_hin     =  input_length_hin
      self.encoder_units        =  encoder_units
      self.decoder_units        =  decoder_units
      self.encoder   = Encoder(vocab_size=self.vocab_size_eng+1,embedding_dim=self.embedding_dim_eng
                               ,encoder_units=self.encoder_units,input_length=self.input_length_eng)
      self.decoder   = Decoder(vocab_size=self.vocab_size_hin+1,embedding_dim=self.embedding_dim_hin
                               ,decoder_units=self.decoder_units,input_length=self.input_length_hin)
      self.dense     = Dense(self.vocab_size_hin, activation='softmax') #

    def call(self,data):
      '''
        A. Pass the input sequence to Encoder layer -- Return encoder_output,encoder_final_state_h,encoder_final_state_c
        B. Pass the target sequence to Decoder layer with intial states as encoder_final_state_h,encoder_final_state_C
        C. Pass the decoder_outputs into Dense layer
        Return decoder_outputs
      '''

      input,output   =   data[0],data[1]
      initial_state  =   self.encoder.initialize_states(batch_size=batch_size)

      encoder_output, encoder_h, encoder_c = self.encoder(input,initial_state)
      states         =   [encoder_h,encoder_c]
      decoder_output,decoder_h, decoder_c =   self.decoder(output, states)
      output          =   self.dense(decoder_output)
      return output

model = Encoder_decoder(vocab_size_eng=vocab_size_eng,vocab_size_hin=vocab_size_hin,
                        embedding_dim_eng=150, embedding_dim_hin=150,
                        input_length_eng=train_dataloader[0][0][0].shape[-1],
                        input_length_hin=train_dataloader[0][0][1].shape[-1],
                        encoder_units=32,
                        decoder_units=32)

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
