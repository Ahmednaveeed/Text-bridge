import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

# Load tokenizers
with open('models/text_tokenizer.pkl', 'rb') as f:
    text_tokenizer = pickle.load(f)

with open('models/summary_tokenizer.pkl', 'rb') as f:
    summary_tokenizer = pickle.load(f)

# Load preprocessed data
from summary_preprocessing import preprocess_summary_data
data = preprocess_summary_data()
(x_train, y_train) = data['train']
(x_val, y_val) = data['val']

# Model parameters
embedding_dim = 128
lstm_units = 256
text_vocab_size = len(text_tokenizer.word_index) + 1
summary_vocab_size = len(summary_tokenizer.word_index) + 1

# Define the model
def create_summarization_model():
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(text_vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(summary_vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(summary_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

# Compile the model
model = create_summarization_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare decoder inputs/targets
decoder_input_data = y_train[:, :-1]
decoder_target_data = y_train[:, 1:]
val_decoder_input = y_val[:, :-1]
val_decoder_target = y_val[:, 1:]

# Model saving callback
checkpoint = ModelCheckpoint(
    'models/summarization_model.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# Train the model
model.fit(
    [x_train, decoder_input_data],
    np.expand_dims(decoder_target_data, -1),
    batch_size=32,
    epochs=20,
    validation_data = (
        [x_val, val_decoder_input],
        np.expand_dims(val_decoder_target, -1)
    )
)
