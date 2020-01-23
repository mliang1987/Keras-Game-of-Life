# Keras-Game-of-Life
 Implementation of Conway's Game of Life with broadcasting.  In addition, created 10x10 Keras model for the game.
- Shichao (Michael) Liang
- mliang@gatech.edu

Files Included:
- conway.py :		Main script for command-line running of the game.
- conway_game.py : 
		Object class for the game.
- pickle_conway.py : 
		Utility methods for pickling and unpickling game board-states
- pickle_tests.py :
		Test cases for pickling utilities
- conway_tests.py :
		Test cases for the main Conway's Game of Life scripts
- conway_keras.py :
		Object class for Keras model for Conway's Game of Life.
		Also includes scripts for generating and querying 10x10 boards.

External Libraries Used:
- numpy
- keras
- tensorflow

To run the game, please use "conway.py" with the following command-line arguments:

    -f [string]: Filepath of a single input instance (.pkl).
    -o [string]: Filepath of a single output instance (.pkl).
    -n [integer]: Number of generations.
    -v [bool]: Verbosity of functions (optional).

Example: 

    python conway.py -f "conway_cl_test1.pkl" -o "conway_out_test1.pkl" -n 3 -v False

To run tests, please just run the relevant test scripts.

    conway_tests.py
    pickle_tests.py

For the Keras Model:
- This is my first time using Keras, and my first time making a neural-network.
- I decided on a Convolutional Neural Network using 2D convolutions since the board
	state is a 2D board.
- For hyper-parameters, I didn't tune many of them, but rather settled on many default values from Keras documentation:
	batch_size = 32 (default)
	epochs = 3 (just to be safe, but for 10x10 2-epochs seems sufficient)
	filters = 32 (default)
	kernel_size = (3,3) as the neighborhood around each cell
	strides = 1 (much simpler than my non-ML implementation!)
	padding = "same" (in order to ensure output matches input)
	activation = "relu" (rectified linear unit) Note: I'm not sure what the differences are, I honestly just used the suggestion in the first resource link below.
- In terms of the neural network architecture, I have three layers:
	1. Conv2D layer for spatial convolution over the board-states
	2. 32-dimension hidden layer (densely-connected)
	3. Output layer (dense) with sigmoidal activation.  
- A model for 10x10 game boards is saved under "conway_keras_model_10x10.h5".  
	The main script in the "conway_keras.py" file trains a model and then queries for 20 iterations to test adherence to ground-truth.
- For training purposes, I created a function that generated n-sized data-sets.  
	For each of these, I used my standard Conway game to calculate ground-truth values for the subsequent generation.
	Then, I determined how to split up the values into training sets, validation sets, and testing sets.  I followed the rule-of-thumb of 60-40 splits, with a final 10% reserved for testing.

Resources Used:
- https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
- https://keras.io/getting-started/sequential-model-guide/
- https://keras.io/layers/core/
- https://keras.io/models/sequential/
- https://keras.io/layers/convolutional/
- https://keras.io/activations/
- https://jovianlin.io/saving-loading-keras-models/
