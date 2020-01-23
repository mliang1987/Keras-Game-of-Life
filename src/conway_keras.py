#!/usr/bin/env python
"""
Conway's Game of Life simulation using numpy and broadcasting.

ConwayGameOfLife holds board state information and functionality to run 
multiple iterations of the game.
"""

import numpy as np
import sys
from conway_game import ConwayGameOfLife
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D

__author__ = "Shichao (Michael) Liang"
__version__ = "1.0"
__maintainer__ = "Shichao (Michael) Liang"
__email__ = "mliang@gatech.edu"

class KerasGameOfLife(object):
    """
    Creates a new Conway's Game of Life object using 2D CNN Sequential Keras model with 
    TensorFlow backend.

    Attributes:
    -------------------
    model : keras.models.Sequential
        Sequential model with a 2D Convolution, hidden Dense layers, one Dense output layer
        with sigmoid activation.
    shape : tuple
        Shape of the game boards

    Methods:
    -------------------
    train_model(training_data, validation_data, batch_size, epochs, suffix)
        Trains the model using training data, validating with validation data, with a 
        specified batch size over a specified number of epochs.
        Saves the model and weights to a file with suffix.
    query(X1, X2)
        Returns True if X2 is one generation subsequent of X1.
    """

    def __init__(self, model, shape):
        """
        Initializes keras model for Conway's Game of Life with default constructor

        Parameters
        -------------------
        model : keras.models.Sequential
            Keras model for CNN for Conway's Game of Life
        shape: tuple
            m x n shape for the boards modeled
        """
        self.model = model
        self.shape = shape
    
    @classmethod
    def from_new(cls, filters, shape, hidden_layer_dims):
        """
        Alternate constructor to construct new model.

        Parameters
        -------------------
        filters : int
            Keras model for CNN for Conway's Game of Life
        shape: tuple
            m x n shape for the boards modeled
        hidden_layer_dims: int
            dimension for the hidden layer in the NN
        """
        model = Sequential()
        model.add(Conv2D(   # First layer: Spatial convolution over images.
            filters = filters, 
            kernel_size = (3,3),
            padding='same',
            activation='relu',
            strides=1,
            input_shape=(shape[0], shape[1], 1)
        ))
        #model.add(Dense(hidden_layer_dims))  # Second layer: Hidden dimensions for middle layer
        model.add(Dense(1, activation = "sigmoid")) # Third (output) layer
        model.compile(loss='binary_crossentropy', 
                        optimizer='rmsprop', 
                        metrics=['accuracy'])
        return cls(model, shape)

    @classmethod
    def from_file(cls, suffix, shape):
        """
        Alternate constructor to read model from file.

        Parameters
        -------------------
        suffix : str
            All models are saved in form "conway_keras_model_{suffix}.h5".
            Providing suffix will load the model from the file.
        shape: tuple
            m x n shape for the boards modeled
        """
        model = load_model("conway_keras_model_{}.h5".format(suffix))
        return cls(model, shape)

    def train_model(self, training_data, validation_data, batch_size = 32, epochs = 2, suffix = ""):
        """
        Trains the model using training data and validation data, saving to a file with
        specified suffix.

        Parameters
        -------------------
        training_data : tuple(nparray)
            training_data[0] contains X
            training_data[1] contains Y
        validation_data : tuple(nparray)
            validation_data[0] contains X
            validation_data[1] contains Y
        batch_size : int
            number of samples per gradient update
        epochs : int
            number of iterations over data during which to train the model.
        suffix : str
            All models are saved in form "conway_keras_model_{suffix}.h5".

        Output
        -------------------
        Saves model to "conway_keras_model_{suffix}.h5".
        """
        self.model.fit(training_data[0], 
                       training_data[1], 
                       batch_size = batch_size, 
                       epochs = epochs, 
                       validation_data = validation_data)
        self.model.save("conway_keras_model_{}.h5".format(suffix))

    def query(self, X1, X2):
        """
        Queries the model to see if X2 is the direct subsequent generation of X1.

        Parameters
        -------------------
        X1 : nparray
            First board state in sequence
        X2 : nparray
            Second board state in sequence

        Returns
        -------------------
        bool 
            Whether or not X2 is direct subsequent generation of X1.
        """
        X1_reshaped = X1.reshape(1, X1.shape[0], X1.shape[1], 1)
        Y_pred = self.model.predict_classes(X1_reshaped)
        Y_flat = Y_pred.reshape(X1.shape[0], X1.shape[1])
        return np.array_equal(Y_flat, X2)

def create_data(n, shape = (5,5)):
    """
    Generates raw data-set for randomly-seeded Game of Life boards of a specified
    shape.

    Parameters
    -------------------
    n : int
        The number of data-sets to generate
    shape : tuple(int, int)
        The shape of the board

    Returns
    -------------------
    X : nparray
        Data-set of boards reshaped for CNN
    Y : nparray
        Corresponding ground-truth for subsequent generation for data-set
    """
    X = np.array([
        np.random.randint(2, size=shape, dtype = np.uint8) 
        for i in range(n)])
    games = np.array([ConwayGameOfLife(board = X_board) for X_board in X])
    for game in games:
        game.run_iterations(1)
    Y = np.array([game.board for game in games])
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    Y = Y.reshape(X.shape)
    return X, Y

def train_new_model(n, shape, suffix):
    """
    Helper function to train a new model given a shape.

    Parameters
    -------------------
    n : int
        The number of data points to generate for training, validation, and testing sets
    shape : tuple(int, int)
        The shape of the board on which to train the model
    suffix : str
        The suffix name of the model
    splits : 3-tuple(int)
        Indices upon which to split the data-set for training, validation, and testing.

    Output
    -------------------
    Saves the model in "conway_keras_model_{suffix}.h5"
    """
    splits = (n, int(n*0.54), int(n*0.9))
    X, Y = create_data(splits[0], shape = shape)
    X_train = X[:splits[1]]
    Y_train = Y[:splits[1]]
    X_val = X[splits[1]:splits[2]]
    Y_val = Y[splits[1]:splits[2]]
    X_test = X[splits[2]:]
    Y_test = Y[splits[2]:]
    keras_game = KerasGameOfLife.from_new(filters = 32, shape = shape, hidden_layer_dims = 32)
    keras_game.train_model((X_train, Y_train), (X_val, Y_val), epochs = 3, suffix = suffix)
    
def query_model(n, shape, suffix):
    """
    Loads model and queries it over a specified number of iterations
    
    Parameters
    -------------------
    n : int
        The number of data points to generate for training, validation, and testing sets
    shape : tuple(int, int)
        The shape of the board on which to train the model
    suffix: str
        The suffix name of the model

    Returns
    -------------------
    bool
        True if all iterations match ground-truth.
    """
    shape = (10,10)
    keras_game = KerasGameOfLife.from_file(suffix, shape)
    match = []
    for i in range(n):
        X1 = np.random.randint(2, size=shape, dtype = np.uint8)
        game = ConwayGameOfLife(board = X1)
        game.run_iterations(1)
        X2 = game.board
        match.append(keras_game.query(X1, X2))
    print("All {} iterations match: {}".format(n, all(match)))
    return all(match)

if __name__ == "__main__":
    train_new_model(n = 100000, shape = (10,10), suffix = "10x10")
    query_model(n = 1000, shape = (10,10), suffix = "10x10")