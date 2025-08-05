import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# load tokenizers
with open('models/en_tokenizer.pkl', 'rb') as f:
    en_tokenizer = pickle.load(f)

with open('models/ur_tokenizer.pkl', 'rb') as f:
    ur_tokenizer = pickle.load(f)

# load preprocessed data
from translation_preprocessing import preprocess_translation_data
(x_train, y_train), (x_val, y_val), _, _ = preprocess_translation_data()

# model parameters
embedding_dim = 256
lstm_units = 512
en_vocab_size = len(en_tokenizer.word_index) + 1
ur_vocab_size = len(ur_tokenizer.word_index) + 1

# define the model
def create_translation_model():
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(en_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(ur_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(ur_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# compile the model
model = create_translation_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare decoder targets (shifted version of y)
decoder_input_data = y_train[:, :-1]
decoder_target_data = y_train[:, 1:]
val_decoder_input = y_val[:, :-1]
val_decoder_target = y_val[:, 1:]

# model saving callback
checkpoint = ModelCheckpoint(
    'models/translation_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose = 1
)

# train model
model.fit(
    [x_train, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=64,
    epochs=20,
    validation_data = (
        [x_val, val_decoder_input],
        np.expand_dims(val_decoder_target, -1)
    )
)