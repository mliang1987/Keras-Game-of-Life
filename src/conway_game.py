#!/usr/bin/env python
"""
Conway's Game of Life simulation using numpy and broadcasting.

ConwayGameOfLife holds board state information and functionality to run 
multiple iterations of the game.
"""

import numpy as np
import sys
from numpy.lib.stride_tricks import as_strided
import pickle_conway as pcw

__author__ = "Shichao (Michael) Liang"
__version__ = "1.0"
__maintainer__ = "Shichao (Michael) Liang"
__email__ = "mliang@gatech.edu"

class ConwayGameOfLife(object):
    """
    Creates a new Conway's Game of Life object.

    Attributes:
    -------------------
    lookup : list(list)
        Lookup-table of values given number of neighbors depending on alive or dead status.  
        First dimension: 0 for dead, 1 for alive.
        Second dimension: 0-8 for number of alive neighbors.
        ex.  lookup[0][3] -> value for a currently dead cell with 3 alive neighbors
    board : nparray 
        m x n representation of the board with 0 or 1s for cell status.
    expanded_board : nparray 
        (m+2) x (n+2) expanded board to handle border cells.

    Methods:
    -------------------
    get_lookup(status, neighbors)
        Returns 0 (dead) or 1 (alive) lookup value depending on cell's current status and 
        number of neighbors.
    get_all_neighbors() -> nparray
        Returns 3x3 slice of board (neighbors) for all cells on the board.
    run_iterations(n : int, verbose = False)
        Runs the game for a specified (n) number of iterations.
        If verbose, will print out the board state each iteration.
    """

    def __init__(self, board = np.random.randint(2, size=(5, 5), dtype = np.uint8)):
        """
        Initializes Conway's Game of Life.

        Parameters
        -------------------
        board : nparray, optional
            At least a 2x2 nparray of 0s and 1s.
            If unspecified, will default to a randomly seeded 5x5 game board.

        Raises
        -------------------
        NotImplementedError
            If the board state is invalid (either not filled with 0s and 1s, or does not
            meet minimum 2x2 size).
        """ 
        # Check for valid filetype for board
        if not isinstance(board, np.ndarray):
            raise NotImplementedError("Board must be an numpy.array.")
        # Check for valid board size.
        if any(x < 2 for x in board.shape):
            raise NotImplementedError("Board state invalid! Must be at least 2x2.")
        # Check for valid board cell entries.
        if not np.all(np.isin(board, [0,1])):
            raise NotImplementedError("Board state invalid! Must be filled with 0s and 1s.")

        # Define lookup table for cell evaluation
        self.lookup = np.asarray([[0,0,0,1,0,0,0,0,0],[0,0,1,1,0,0,0,0,0]])

        # Define expanded board for easier border cell calculations
        expanded_shape = tuple(d+2 for d in board.shape)
        board_slice = (slice(1, -1),) * 2
        self.expanded_board = np.zeros(expanded_shape,dtype = np.uint8)
        self.expanded_board[board_slice] = board
        self.board = self.expanded_board[board_slice]

    def get_lookup(self, cell_status, num_neighbors):
        """
        Returns next cell state depending on current cell state and number of alive neighbors.

        Parameters
        -------------------
        cell_status : int
            0 for currently dead; 1 for currently alive
        num_neighbors : int
            The number of alive neighbors (0-8)

        Returns
        -------------------
        status : int
            Cell status for next iteration
            0 for dead; 1 for alive
        """
        return self.lookup[cell_status,num_neighbors]

    def get_all_neighbors(self):
        """
        Returns all 3x3 slices of the board corresponding to neighbors for all cells on the board.

        Returns:
        -------------------
        all_neighbors : (m,n,3,3)-shaped nparray
            For a specified cell at location (r,c), all_neighbors[r][c] is the 3x3 subgrid of the
            board centered around cell (r,c).
        """
        m, n = self.board.shape
        return as_strided(self.expanded_board,
                          shape = (m,n,3,3), 
                          strides = self.expanded_board.strides + self.expanded_board.strides)
    
    def run_iterations(self, n, verbose = False):
        """
        Runs the game for a specified (n) number of iterations.

        Parameters:
        -------------------
        n : int
            The number of iterations to run the game.
        verbose : bool, optional
            If verbose, will print out the board state each iteration.
        """
        for i in range(n):
            # Calculate total number of neighbors for each cell
            all_neighbors = self.get_all_neighbors()
            all_num_neighbors = np.sum(all_neighbors, axis = (-2,-1)) - self.board
            # Determine new state for each cell using lookup table and number of neighbors
            self.board[:] = np.where(self.board, 
                                     self.lookup[1][all_num_neighbors], 
                                     self.lookup[0][all_num_neighbors])
            # Verbosity check
            if verbose:
                print(self.board)

if __name__ == "__main__":
    read_grid = np.array([[0,0,0,0,0],[0,1,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
    print(read_grid)
    game = ConwayGameOfLife(board = read_grid)
    game.run_iterations(4)
    print(game.board)