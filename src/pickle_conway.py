#!/usr/bin/env python
"""
Pickling and unpickling of board states for Conway's Game of Life.

Assumes board state is represented as nparray of 0s and 1s.
"""

import numpy as np
import datetime as dt
import sys
import pickle

__author__ = "Shichao (Michael) Liang"
__version__ = "1.0"
__maintainer__ = "Shichao (Michael) Liang"
__email__ = "mliang@gatech.edu"

def pickle_conway_board(board_state: np.ndarray, filename = None, verbose = False) -> bool:
    """
    Pickles a conway board state (numpy array)
    
    Parameters:
    -------------------
    - board_state (nparray) - Numpy array representing the board state.
    - filename (str) - Name of the file in which to pickle the board state. Defaults
        to 'conwayboard'.

    Returns success state (boolean) for pickling with check for type.
    """
    # Returns False if input is invalid type.
    if not isinstance(board_state, np.ndarray):
        if verbose: print("Invalid type. Could not save board-state to file:",filename)
        return False

    # If no filename provided, creates default filename with timestamp.
    if not filename:
        filename = "conway{}.pkl".format(dt.datetime.now().strftime("%d%m%Y.%H-%M-%S"))
        if verbose: print(filename)

    # Writes board-state to file.
    with open(filename, 'wb') as outfile:
        pickle.dump(board_state, outfile)
        return True

def unpickle_conway_board(filename, verbose = False):
    """
    Unpickles a conway board state (numpy array)

    Parameters:
    - filename (str) - Filename in which to read the board state without extension.

    Returns:
    - Output (nparray) - Board state of the Game of Life saved or None if unable to 
        read board-state due to not finding a valid file.
    """
    try:
        # Unpickles file
        with open(filename,"rb") as unpickle:
            conway_board = pickle.load(unpickle)

        # Checks to see if unpickled data structure is indeed an nparray
        if isinstance(conway_board, np.ndarray):
            return conway_board
        else:
            # Unpickled file does not contain nparray
            if verbose: print("Could not read valid Game of Life board-state.")
            return None

    # Handles IO Exception with no valid file found
    except OSError:
        if verbose: print("Could not open/read file:", filename)
        return None