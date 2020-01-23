#!/usr/bin/env python
"""
Conway's Game of Life test cases.

See individual cases for test descriptions.
"""

import unittest
import numpy as np
from conway_game import ConwayGameOfLife

__author__ = "Shichao (Michael) Liang"
__version__ = "1.0"
__maintainer__ = "Shichao (Michael) Liang"
__email__ = "mliang@gatech.edu"

class TestConway(unittest.TestCase):
    """
    Class for unit tests for Conway's Game of Life.

    See individual test cases for test descriptions.
    """

    def test_conway(self):
        """
        Test for proper construction of game object.
        """
        read_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = ConwayGameOfLife(board = read_grid)
        test_grid = game.board
        np.testing.assert_equal(test_grid, read_grid)

    def test_conway_alive_rules(self):
        """
        Test for proper adherence to game of life rules for currently alive cells.
        """
        read_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = ConwayGameOfLife(board = read_grid)
        alive_list = [game.get_lookup(1, n) for n in range(9)]
        self.assertListEqual(alive_list, [0,0,1,1,0,0,0,0,0])

    def test_conway_dead_rules(self):
        """
        Test for proper adherence to game of life rules for currently dead cells.
        """
        read_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = ConwayGameOfLife(board = read_grid)
        dead_list = [game.get_lookup(0,n) for n in range(9)]
        self.assertListEqual(dead_list, [0,0,0,1,0,0,0,0,0])

    def test_conway_too_small(self):
        """
        Test that constructor throws error when given invalid size board-state
        """
        read_grid = np.array([[0, 0]])
        with self.assertRaises(NotImplementedError):
            game = ConwayGameOfLife(board = read_grid)

    def test_conway_invalid_values(self):
        """
        Test for constructor throwing error when given invalid size board-state
        """
        read_grid = np.array([[0, 1, 2], [0, 1, 0], [0, 1, 0]])
        with self.assertRaises(NotImplementedError):
            game = ConwayGameOfLife(board = read_grid)

    def test_conway_invalid_type(self):
        """
        Test for constructor throwing error when given invalid size board-state
        """
        read_grid = 3
        with self.assertRaises(NotImplementedError):
            game = ConwayGameOfLife(board = read_grid)

    def test_conway_oscillator1(self):
        """
        Test for 3x3 oscillator flipping once
        """
        read_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(1)
        test_grid = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        np.testing.assert_equal(test_grid, game.board)

    def test_conway_oscillator2(self):
        """
        Test for 3x3 oscillator flipping twice
        """
        read_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(2)
        test_grid = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        np.testing.assert_equal(test_grid, game.board)

    def test_conway_boat_still(self):
        """
        Test for boat still pattern:
            110
            101
            010
        """
        read_grid = np.array([[0,0,0,0,0],[0,1,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(1)
        test_grid = np.array([[0,0,0,0,0],[0,1,1,0,0],[0,1,0,1,0],[0,0,1,0,0],[0,0,0,0,0]])
        np.testing.assert_equal(game.board, test_grid)

    def test_conway_glider1(self):
        """
        Test for single iteration glider pattern.
            001
            101
            011
        """
        read_grid = np.array([[0,0,0,0,0],[0,0,1,0,0],[1,0,1,0,0],[0,1,1,0,0],[0,0,0,0,0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(1)
        test_grid = np.array([[0,0,0,0,0],[0,1,0,0,0],[0,0,1,1,0],[0,1,1,0,0],[0,0,0,0,0]])
        np.testing.assert_equal(game.board, test_grid)

    def test_conway_glider2(self):
        """
        Test for 2-iteration glider pattern.
            001
            101
            011
        """
        read_grid = np.array([[0,0,0,0,0],[0,0,1,0,0],[1,0,1,0,0],[0,1,1,0,0],[0,0,0,0,0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(2)
        test_grid = np.array([[0,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0]])
        np.testing.assert_equal(game.board, test_grid)

    def test_conway_glider3(self):
        """
        Test for 2-iteration glider pattern.
            001
            101
            011
        """
        read_grid = np.array([[0,0,0,0,0],[0,0,1,0,0],[1,0,1,0,0],[0,1,1,0,0],[0,0,0,0,0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(3)
        test_grid = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,0,1,0],[0,0,1,1,0],[0,0,1,0,0]])
        np.testing.assert_equal(game.board, test_grid)

    def test_conway_glider4(self):
        """
        Test for 2-iteration glider pattern.
            001
            101
            011
        """
        read_grid = np.array([[0,0,0,0,0],[0,0,1,0,0],[1,0,1,0,0],[0,1,1,0,0],[0,0,0,0,0]])
        game = ConwayGameOfLife(board = read_grid)
        game.run_iterations(4)
        test_grid = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,1,0],[0,1,0,1,0],[0,0,1,1,0]])
        np.testing.assert_equal(game.board, test_grid)

if __name__ == "__main__":
    unittest.main()