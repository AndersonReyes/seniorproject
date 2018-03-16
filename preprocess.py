import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence


categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
              'identity_hate']

vocab_size = 20000
max_seqlength = 500
embedding_size = 128

data = pd.read_csv('data/train.csv')
del data['id']
labels = data[categories].values.tolist()
sentences = data[['comment_text']].values
sentences = np.squeeze(sentences)


with open('data/sentences.npy', 'wb') as f:
    np.save(f, sentences)
    
with open('data/labels.npy', 'wb') as f:
    np.save(f, labels)


tokenizer = Tokenizer(num_words=vocab_size)
# fit the tokenizer to the sentences
tokenizer.fit_on_texts(sentences)
seq_matrix = tokenizer.texts_to_matrix(sentences)
sentences = sequence.pad_sequences(seq_matrix, maxlen=max_seqlength)

with open('data/processed_sentences.npy', 'wb') as f:
    np.save(f, sentences)

