import os 
import numpy as np
import pandas as pd
from tensorflo.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# paths
train_path = 'data/summarization/train.csv'
val_path = 'data/summarization/val.csv'
test_path = 'data/summarization/test.csv'

# load data 
def load_summary_data():
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df

# clean text
def clean_text(sentences):
    return[str(s).lower().strip() for s in sentences]

# tokenize and padding
def tokenize_pad(sentences, lang, num_words=200000, max_len=300):
    tokenizer = Tokenizer(num_words = num_words, oov_token='<OOV>', filter = '')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post', maxlen=max_len, truncating='post')

    print(f"[{lang}] Vocab Size: {len(tokenizer.word_index) + 1}")
    print(f"[{lang}] Example Sequence: {sequences[0][:10]}")
    return tokenizer, padded

# preprocess data
def preprocess_summary_data():
    train_df, val_df, test_data = load_summary_data()

    train_texts = clean_text(train_df['text'].tolist())
    train_summaries = clean_text(train_df['summary'].tolist())

    val_texts = clean_text(val_df['text'].tolist())
    val_summaries = clean_text(val_df['summary'].tolist())

    test_texts = clean_text(test_data['text'].tolist())
    test_summaries = clean_text(test_data['summary'].tolist())

    text_tokenizer , train_text_seq = tokenize_pad(train_texts, 'input text', max_len=300, num_words=200000)
    summary_tokenizer, train_summary_seq = tokenize_pad(train_summaries, 'summary', max_len=60, num_words=10000)

    val_text_seq = tokenize_pad(text_tokenizer.texts_to_sequences(val_texts), padding='post', maxlen=300, truncating='post')
    val_summary_seq = tokenize_pad(summary_tokenizer.texts_to_sequences(val_summaries), padding='post', maxlen=60, truncating='post')

    test_text_seq = tokenize_pad(text_tokenizer.texts_to_sequences(test_texts), padding='post', maxlen=300, truncating='post')
    test_summary_seq = tokenize_pad(summary_tokenizer.texts_to_sequences(test_summaries), padding='post', maxlen=60, truncating='post')

    return {
        "train": (train_text_seq, train_summary_seq),
        "val": (val_text_seq, val_summary_seq),
        "test": (test_text_seq, test_summary_seq),
        "text_tokenizer": text_tokenizer,
        "summary_tokenizer": summary_tokenizer
    }