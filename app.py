import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load tokenizers
with open('models/en_tokenizer.pkl', 'rb') as f:
    en_tokenizer = pickle.load(f)

with open('models/ur_tokenizer.pkl', 'rb') as f:
    ur_tokenizer = pickle.load(f)

with open('models/text_tokenizer.pkl', 'rb') as f:
    text_tokenizer = pickle.load(f)

with open('models/summary_tokenizer.pkl', 'rb') as f:
    summary_tokenizer = pickle.load(f)

# Load models
translation_model = tf.keras.models.load_model('models/translation_model.keras')
summarization_model = tf.keras.models.load_model('models/summarization_model.keras')

# Constants
max_len_en = 50
max_len_ur = 50
max_len_text = 300
max_len_summary = 50

# Utility functions
def preprocess_input(text, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def decode_sequence(pred_seq, tokenizer):
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    result = []
    for idx in pred_seq:
        if idx == 0:
            continue
        word = index_word.get(idx, '')
        if word == '<eos>' or word == '':
            break
        result.append(word)
    return ' '.join(result)

def predict_translation(text):
    encoder_input = preprocess_input(text, en_tokenizer, max_len_en)
    decoder_input = np.zeros((1, max_len_ur))
    decoder_input[0, 0] = ur_tokenizer.word_index.get('<sos>', 1)

    for i in range(1, max_len_ur):
        preds = translation_model.predict([encoder_input, decoder_input], verbose=0)
        next_word = np.argmax(preds[0, i-1])
        decoder_input[0, i] = next_word
        if next_word == ur_tokenizer.word_index.get('<eos>', 2):
            break

    return decode_sequence(decoder_input[0], ur_tokenizer)

def predict_summary(text):
    encoder_input = preprocess_input(text, text_tokenizer, max_len_text)
    decoder_input = np.zeros((1, max_len_summary))
    decoder_input[0, 0] = summary_tokenizer.word_index.get('<sos>', 1)

    for i in range(1, max_len_summary):
        preds = summarization_model.predict([encoder_input, decoder_input], verbose=0)
        next_word = np.argmax(preds[0, i-1])
        decoder_input[0, i] = next_word
        if next_word == summary_tokenizer.word_index.get('<eos>', 2):
            break

    return decode_sequence(decoder_input[0], summary_tokenizer)

# Streamlit UI
st.title("🧠 English to Urdu Translator & Summarizer")

text = st.text_area("Enter English text:")

if st.button("Translate to Urdu"):
    if text.strip():
        urdu_translation = predict_translation(text)
        st.markdown("### 🈶 Urdu Translation")
        st.success(urdu_translation)
    else:
        st.warning("Please enter some text first.")

if st.button("Summarize Text"):
    if text.strip():
        summary = predict_summary(text)
        st.markdown("### ✂️ English Summary")
        st.info(summary)
    else:
        st.warning("Please enter some text first.")
