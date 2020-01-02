from __future__ import print_function

from keras.models import Model
from keras.models import load_model
from keras.layers import Embedding
from keras.layers import Input, LSTM, Dense
from keras import optimizers
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

batch_size = 128  # Batch size for training.
epochs = 3  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding spac
num_samples = 1000  # Number of samples to train on.
embedding_dim = 300
# Path to the data txt file on disk.
data_path = 'data/clean_train.txt'
val_data_path = 'data/clean_validation.txt'

# Vectorize the data.
input_texts = []
target_texts = []
val_input_texts = []
val_target_texts = []
input_characters = set()
input_characters.add('<unk>')
target_characters = set()
target_characters.add('<unk>')

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
with open(val_data_path, 'r', encoding='utf-8') as f:
    val_lines = f.read().split('\n')

# train data
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    # word base
    target_text = word_tokenize(target_text)
    target_text.insert(0, '<start>')
    target_text.append('<end>')
    input_text = word_tokenize(input_text)

    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

# validation data
for line in val_lines[: min(num_samples, len(val_lines) - 1)]:
    val_input_text, val_target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    # word_base
    val_target_text = word_tokenize(val_target_text)
    val_target_text.insert(0, '<start>')
    val_target_text.append('<end>')
    val_input_text = word_tokenize(val_input_text)

    val_input_texts.append(val_input_text)
    val_target_texts.append(val_target_text)

texts = []
for t in input_texts:
    texts += t
freq = nltk.FreqDist(texts)
for key, val in freq.items():
    if val == 1:
        input_characters.remove(key)

texts = []
for t in target_texts:
    texts += t
freq = nltk.FreqDist(texts)
for key, val in freq.items():
    if val == 1:
        target_characters.remove(key)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
input_characters.insert(0,'<pad>')
target_characters.insert(0,'<pad>')
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype='int')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype='int')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, 1), dtype='int')

val_encoder_input_data = np.zeros((len(val_input_texts), max_encoder_seq_length), dtype='int')
val_decoder_input_data = np.zeros((len(val_input_texts), max_decoder_seq_length), dtype='int')
val_decoder_target_data = np.zeros((len(val_input_texts), max_decoder_seq_length, 1), dtype='int')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        if char in input_token_index:
            encoder_input_data[i, t] = input_token_index[char]
        else:
            encoder_input_data[i, t] = input_token_index['<unk>']

    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if char in target_token_index:
            index = target_token_index[char]
        else:
            index = target_token_index['<unk>']
        decoder_input_data[i, t] = index
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, 0] = index

for i, (val_input_text, val_target_text) in enumerate(zip(val_input_texts, val_target_texts)):
    for t, char in enumerate(val_input_text):
        if t >= max_encoder_seq_length:
            break

        if char in input_token_index:
            val_encoder_input_data[i, t] = input_token_index[char]
        else:
            val_encoder_input_data[i, t] = input_token_index['<unk>']

    for t, char in enumerate(val_target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        if t >= max_decoder_seq_length:
            break

        if char in target_token_index:
            index = target_token_index[char]
        else:
            index = target_token_index['<unk>']

        val_decoder_input_data[i, t] = index
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            val_decoder_target_data[i, t - 1, 0] = index

# embedding layer
word2vec_file = "GoogleNews-vectors-negative300.bin"
word2vec = KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
# for encoder
embedding_matrix = np.zeros((len(input_characters) + 1, embedding_dim))
for i, word in enumerate(input_characters):
    if word in word2vec:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = word2vec.wv[word]
# for decoder
de_embedding_matrix = np.zeros((len(target_characters) + 1, embedding_dim))
for i, word in enumerate(target_characters):
    if word in word2vec:
        # words not found in embedding index will be all-zeros.
        de_embedding_matrix[i] = word2vec.wv[word]

encoder_inputs = Input(shape=(None,))
embedding_layer = Embedding(len(input_characters) + 1, embedding_dim,
                            weights=[embedding_matrix],
                            input_length=None,
                            trainable=True,
                            mask_zero=True)
embedding_encoder_inputs = embedding_layer(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(embedding_encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
de_embedding_layer = Embedding(len(target_characters) + 1, embedding_dim,
                            weights=[de_embedding_matrix],
                            input_length=None,
                            trainable=True,
                            mask_zero=True)
embedding_decoder_inputs = de_embedding_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(embedding_decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
# adam = optimizers.Adam(lr=0.001, epsilon=1e-08)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.summary()
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data))
# Save model
model.save('s2s.h5')

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    embedding_decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # print(states_value.shape)
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['<start>']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_char = target_characters[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<end>' or
           len(decoded_sentence) >= max_decoder_seq_length):
            stop_condition = True
        else:
            # Update the target sequence (of length 1).
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

    return decoded_sentence


for seq_index in range(5):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('---')
    print('Input sentence:', input_texts[seq_index])
    print('')
    print('Decoded sentence:', decoded_sentence)

