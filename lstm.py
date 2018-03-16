import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers.embeddings import Embedding

class LSTMRNN:
    """
    Long-Term Short-Term RNN

    Args:
        :lstm_units: number of units for lstm cell
        :embedding_size: size of the initial embedding layer
        :vocab_size: size of the vocabulary
        :input_size: size of the input sentences
        :output_size: number of classification classes
        :activation: activation function for conv layers

    """
    def __init__(self, vocab_size, embedding_size, input_size, lstm_units,
                 output_size, activation='relu'):
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.lstm_units = lstm_units
        self.model = None

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_size, input_length=self.input_size))
        self.model.add(LSTM(self.lstm_units))
        self.model.add(Dense(self.output_size, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print(self.model.summary())

    def fit(self, X, Y, X_test, Y_test, batch_size=256, epochs=20, savename='cnn'):
        self.model.fit(X, Y, validation_data=(X_test, Y_test),
                       batch_size=batch_size, epochs=epochs)
        
        # save the model
        # serialize model to JSON
        model_json = self.model.to_json()
        with open('./models/{}.json'.format(savename), 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights('./models/{}.h5'.format(savename))

    def predict(self, X, Y):
        return self.model.predict(X, Y)
    
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)