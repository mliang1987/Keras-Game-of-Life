#!/usr/bin/env python
"""
Pickling and unpickling test cases.

See individual test cases for test descriptions.
"""

import unittest
import numpy as np
import pickle
import pickle_conway as pcw

__author__ = "Shichao (Michael) Liang"
__version__ = "1.0"
__maintainer__ = "Shichao (Michael) Liang"
__email__ = "mliang@gatech.edu"

class TestPickleConway(unittest.TestCase):
    """
    Class for unit tests for pickling/unpickling Conway's Game of Life board-state.

    See individual test cases for test descriptions.
    """

    def test_pickle(self):
        """
        Picking Test 1: Assesses if the pickling method successfully saves a pickle file with
        provided filename and board-state (nparray)
        """
        test_grid = np.zeros((3,3))
        test_result = pcw.pickle_conway_board(test_grid, filename = "conway_pickle_test1.pkl")
        self.assertEqual(test_result, True, "Pickling result of 3x3 nparray of zeroes should be True.")

    def test_pickle_save_invalid_type(self):
        """
        Picking Test 2: Assesses if the pickling method successfully identifies an invalid 
        datatype for the board-state and returns a failure to pickle the data structure.
        """
        test_grid = 3
        test_result = pcw.pickle_conway_board(test_grid, filename = "conway_pickle_test2.pkl")
        self.assertEqual(test_result, False, "Pickling result of int should be False.")

    def test_pickle_save_nofilename(self):
        """
        Picking Test 3: Assesses if the pickling method successfully saves a pickle file without
        provided filename.
        """
        test_grid = np.zeros((4,4))
        test_result = pcw.pickle_conway_board(test_grid)
        self.assertEqual(test_result, True, "Pickling result of 4x4 nparray with no filename should be True")

    def test_unpickle(self):
        """
        Unpickling Test 1: Assesses if the unpickling method successfully returns a board-state 
        (nparray) given a provided filename.
        """
        test_grid = np.zeros((3,3))
        test_result = pcw.pickle_conway_board(test_grid, filename = "conway_unpickle_test1.pkl")
        read_grid = pcw.unpickle_conway_board(filename = "conway_unpickle_test1.pkl")
        np.testing.assert_equal(test_grid, read_grid)

    def test_unpickle_invalid_type(self):
        """
        Unpickling Test 2: Assesses if the unpickling method successfully identifies a pickle file
        with an invalid data structure for the board-state and returns None.
        """
        test_grid = 3
        with open("conway_unpickle_test2.pkl", 'wb') as outfile:
            pickle.dump(test_grid, outfile)
        test_result = pcw.unpickle_conway_board(filename = "conway_unpickle_test2.pkl")
        self.assertEqual(test_result, None, "Unpickling non-nparray object results in None")

    def test_unpickle_nofile(self):
        """
        Unpickling Test 3: Assesses if the unpickling method sucessfully identifies a non-existent
        pickle file and returns None as a result.
        """
        test_result = pcw.unpickle_conway_board(filename = "conway_unpickle_test3.pkl")
        self.assertEqual(test_result, None, "Unpickling non-existent file results in None")

if __name__ == "__main__":
    unittest.main()