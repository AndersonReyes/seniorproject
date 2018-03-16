import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPool1D, concatenate
from keras.layers import Dropout, Flatten, Input
from keras.models import Model
from keras.layers.embeddings import Embedding

class TextCNN:

    """
    Text classification class using Convolutional neural networks

    Args:
        :nfilters: number of filters
        :filter_sizes: list of kernel size for each filter
        :embedding_size: size of the initial embedding layer
        :vocab_size: size of the vocabulary
        :input_size: size of the input sentences
        :output_size: number of classification classes
        :activation: activation function for conv layers

    """
    def __init__(self, nfilters, filter_sizes, embedding_size, vocab_size,
                 input_size, output_size, activation='relu'):
        self.nfilters = nfilters
        self.filter_sizes = filter_sizes
        self.activation = activation
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.model = None

    def build_model(self):
        seq_input = Input(shape=(self.input_size, ), dtype='int32')
        self.model = Embedding(self.vocab_size, self.embedding_size,
                               input_length=self.input_size)(seq_input)
        
        # Convolution outputs in parallel will be using the input
        conv_outputs = []
        
        for size in self.filter_sizes:
            sub_model = Conv1D(self.nfilters, size, activation=self.activation)(self.model)
            sub_model = MaxPool1D(pool_size=size)(sub_model)
            conv_outputs.append(sub_model)
        
        # Concatenate layers
        merged = concatenate(conv_outputs, axis=1)
        self.model = Flatten()(merged)
        self.model = Dropout(0.5)(self.model)
        self.model = Dense(self.output_size, activation='sigmoid')(self.model)
        self.model = Model(seq_input, self.model)
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop',
                           metrics=['accuracy'])
        
        print(self.model.summary())

    def fit(self, X, Y, X_test, Y_test, batch_size=256, epochs=20, savename='lstm'):
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
    

if __name__ == '__main__':
    test_model = TextCNN(
        nfilters=1, filter_sizes=[1, 2], embedding_size=12, vocab_size=5,
        input_size=5, output_size=1)
    test_model.build_model()
    test_model.fit(np.array([[1, 1, 2, 3, 4]]), np.array([[1]]), np.array([[1, 1, 2, 3, 4]]), np.array([[1]]), epochs=1)