import numpy as pd
import numpy as np
from text_cnn import TextCNN
from lstm import LSTMRNN
from preprocess import vocab_size, max_seqlength, categories, embedding_size
from sklearn.model_selection import train_test_split

X = np.load('data/processed_sentences.npy')
y = np.load('data/labels.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20,
                                                    random_state=42)

def train_cnn():
    model_cnn = TextCNN(128, [2, 3, 5], embedding_size, vocab_size, max_seqlength,
                        len(categories))

    model_cnn.build_model()
    model_cnn.fit(X_train, y_train, X_test, y_test, epochs=20,
                  savename='cnn_128_2.3.5_128_500')

def train_lstm():
    model_lstm = LSTMRNN(vocab_size, embedding_size, max_seqlength, 200, len(categories))
    model_lstm.build_model()
    model_lstm.fit(X_train, y_train, X_test, y_test, epochs=2,
                   savename='lstm_128_500_200')

if __name__ == '__main__':
    train_cnn()

