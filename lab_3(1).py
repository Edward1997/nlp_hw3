from __future__ import print_function

from keras.models import Model
from keras.layers import Embedding
from keras.layers import Input, LSTM, Dense
from keras import optimizers
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

batch_size = 32  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
embedding_dim = 300
# Path to the data txt file on disk.
data_path = 'data/train.txt'
val_data_path = 'data/validation.txt'

# Vectorize the data.
input_texts = []
target_texts = []
val_input_texts = []
val_target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
with open(val_data_path, 'r', encoding='utf-8') as f:
    val_lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    # word base
    target_text = target_text.split()
    target_text = [''.join(list(filter(str.isalpha, word))) for word in target_text]
    target_text = list(filter(None, target_text))
    target_text.insert(0,'\t')
    target_text.append('\n')
    input_text = input_text.split()
    input_text = [''.join(list(filter(str.isalpha, word))) for word in input_text]
    input_text = list(filter(None, input_text))
    # char base
    # target_text = ''.join(list(filter(str.isalpha, target_text)))
    # target_text = '\t' + target_text + '\n'
    # input_text = ''.join(list(filter(str.isalpha, input_text)))

    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
for line in val_lines[: min(num_samples, len(val_lines) - 1)]:
    val_input_text, val_target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    # word_base
    val_target_text = val_target_text.split()
    val_target_text = [''.join(list(filter(str.isalpha, word))) for word in val_target_text]
    val_target_text = list(filter(None, val_target_text))
    val_target_text.insert(0,'\t')
    val_target_text.append('\n')
    val_input_text = val_input_text.split()
    val_input_text = [''.join(list(filter(str.isalpha, word))) for word in val_input_text]
    val_input_text = list(filter(None, val_input_text))
    # char base
    # val_target_text = ''.join(list(filter(str.isalpha, val_target_text)))
    # val_target_text = '\t' + val_target_text + '\n'
    # val_input_text = ''.join(list(filter(str.isalpha, val_input_text)))

    val_input_texts.append(val_input_text)
    val_target_texts.append(val_target_text)
    for char in val_input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in val_target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max(max([len(txt) for txt in input_texts]), max([len(txt) for txt in val_input_texts]))
max_decoder_seq_length = max(max([len(txt) for txt in target_texts]), max([len(txt) for txt in val_target_texts]))

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
        encoder_input_data[i, t] = input_token_index[char]

    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        index = target_token_index[char]
        decoder_input_data[i, t] = index
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, 0] = index

for i, (val_input_text, val_target_text) in enumerate(zip(val_input_texts, val_target_texts)):
    for t, char in enumerate(val_input_text):
        val_encoder_input_data[i, t] = input_token_index[char]

    for t, char in enumerate(val_target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        index = target_token_index[char]
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

    # Define an input sequence and process it.

# print(embedding_matrix[encoder_input_data[1, 3]])
encoder_inputs = Input(shape=(max_encoder_seq_length,))
embedding_layer = Embedding(len(input_characters) + 1, embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_encoder_seq_length,
                            trainable=False)
embedding_encoder_inputs = embedding_layer(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(embedding_encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(max_decoder_seq_length,))
de_embedding_layer = Embedding(len(target_characters) + 1, embedding_dim,
                            weights=[de_embedding_matrix],
                            input_length=max_decoder_seq_length,
                            trainable=False)
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


def data_generator(batch_size, encoder_input_data, decoder_input_data, decoder_target_data):
    total_size = len(encoder_input_data)
    batch_num = total_size // batch_size
    while 1:
        batch_id = 0
        while batch_id < batch_num:
            yield [encoder_input_data[batch_id * batch_size:(batch_id + 1) * batch_size - 1],
                   decoder_input_data[batch_id * batch_size:(batch_id + 1) * batch_size - 1]], decoder_target_data[batch_id * batch_size:(batch_id + 1) * batch_size - 1]
            batch_id += 1
            
        yield [encoder_input_data[batch_id * batch_size:],
               decoder_input_data[batch_id * batch_size:]], decoder_target_data[batch_id * batch_size:]

    return


# Run training
adam = optimizers.Adam(lr=0.0001, epsilon=1e-08)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.summary()
model.fit_generator(data_generator(batch_size, encoder_input_data, decoder_input_data, decoder_target_data),
                    steps_per_epoch=len(encoder_input_data)//batch_size + 1,
                    epochs=epochs,
                    validation_data=data_generator(batch_size, val_encoder_input_data, val_decoder_input_data,
                                                   val_decoder_target_data),
                    validation_steps=len(val_encoder_input_data)//batch_size + 1)

# Save model
model.save('s2s.h5')



