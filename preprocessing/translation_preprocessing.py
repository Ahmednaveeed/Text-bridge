import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# File paths
eng_path = 'data/translation/english.txt'
urdu_path = 'data/translation/urdu.txt'

# load data
def load_translation_data():
    with open(eng_path, 'r', encoding='utf-8') as f:
        eng_lines = f.read().strip().split('\n')
    with open(urdu_path, 'r', encoding='utf-8') as f:
        urdu_lines = f.read().strip().split('\n')

    return eng_lines, urdu_lines

# clean text
def clean_text(sentences):
    return [s.strip().lower() for s in sentences]

# tokenize and padding
def tokenize(sentences, lang, num_words=200000):
    tokenizer = Tokenizer(num_words = num_words, filter='', oov_token = '<OOV>')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')
    print(f"[{lang}] Vocab Size: {len(tokenizer.word_index) + 1}")
    return padded_sequences, tokenizer

# preprocess data
def preprocess_translation_data():
    en, ur = load_translation_data()
    en, ur = clean_text(en), clean_text(ur)

    en_tokenizer, en_padded = tokenize(en, 'English')
    ur_tokenizer, ur_padded = tokenize(ur, 'Urdu')

    # Save English tokenizer
    with open('models/en_tokenizer.pkl', 'wb') as f:
        pickle.dump(en_tokenizer, f)

    # Save Urdu tokenizer
    with open('models/ur_tokenizer.pkl', 'wb') as f:
        pickle.dump(ur_tokenizer, f)

    x_train, x_test, y_train, y_test = train_test_split(en_padded, ur_padded, test_size=0.2, random_state=42)