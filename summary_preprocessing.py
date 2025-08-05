import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# paths
train_path = 'data/summarization/train.csv'
val_path = 'data/summarization/validation.csv'
test_path = 'data/summarization/test.csv'

# load data
def load_summary_data():
    train_df = pd.read_csv(train_path, engine='python')
    val_df = pd.read_csv(val_path, engine='python')
    test_df = pd.read_csv(test_path, engine='python')

    '''
    train_df = train_df.iloc[:50000]
    val_df = val_df.iloc[:6000]
    test_df = test_df.iloc[:6000]
    '''
    return train_df, val_df, test_df

# clean text
def clean_text(sentences):
    return[str(s).lower().strip() for s in sentences]

# tokenize and padding
def tokenize_pad(sentences, lang, num_words=50000, max_len=200):
    tokenizer = Tokenizer(num_words = num_words, oov_token='<OOV>', filters = '')
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, padding='post', maxlen=max_len, truncating='post')

    print(f"[{lang}] Vocab Size: {len(tokenizer.word_index) + 1}")
    print(f"[{lang}] Example Sequence: {sequences[0][:10]}")
    return tokenizer, padded

# preprocess data
def preprocess_summary_data():
    train_df, val_df, test_data = load_summary_data()

    train_texts = clean_text(train_df['article'].tolist())
    train_summaries = clean_text(train_df['highlights'].tolist())

    val_texts = clean_text(val_df['article'].tolist())
    val_summaries = clean_text(val_df['highlights'].tolist())

    test_texts = clean_text(test_data['article'].tolist())
    test_summaries = clean_text(test_data['highlights'].tolist())


    text_tokenizer , train_text_seq = tokenize_pad(train_texts, 'input text', max_len=200, num_words=50000)
    summary_tokenizer, train_summary_seq = tokenize_pad(train_summaries, 'summary', max_len=40, num_words=8000)

    # Save tokenizers for later use
    with open('models/text_tokenizer.pkl', 'wb') as f:
        pickle.dump(text_tokenizer, f)
    with open('models/summary_tokenizer.pkl', 'wb') as f:
        pickle.dump(summary_tokenizer, f)

    # Split data into training and validation sets
    x_train, x_temp, y_train, y_temp = train_test_split(train_text_seq, train_summary_seq, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


    return {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test),
        'text_tokenizer': text_tokenizer,
        'summary_tokenizer': summary_tokenizer
    }

if __name__ == "__main__":
    preprocess_summary_data()