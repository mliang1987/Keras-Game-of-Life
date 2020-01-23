#!/usr/bin/env python
"""
Main script for running Conway's Game of Life simulation.

Command-line arguments:
  -f [string]: Filepath of a single input instance.
  -o [string]: Filepath of a single output instance.
  -n [integer]: Number of generations.
  -v [bool]: Verbosity of functions (optional).
"""

import numpy as np
import sys
import pickle_conway as pcw
from conway_game import ConwayGameOfLife

__author__ = "Shichao (Michael) Liang"
__version__ = "1.0"
__maintainer__ = "Shichao (Michael) Liang"
__email__ = "mliang@gatech.edu"

def main():
    """
    Command-line runtime for Conway's game of life.

    Command-line arguments:
      -f [string]: Filepath of a single input instance.
      -o [string]: Filepath of a single output instance.
      -n [integer]: Number of generations.
      -v [bool]: Verbosity of functions (optional).
    """

    # Parse command-line arguments.
    cli_args = dict((sys.argv[1 + i], sys.argv[2 + i]) for i in range(0, len(sys.argv[1:]), 2))
    input_arg = cli_args.get("-f", None)
    output_arg = cli_args.get("-o", None)
    generation_arg = cli_args.get("-n", None)
    verbose_arg = cli_args.get("-v", False) == "True"
    if not input_arg or not output_arg or not generation_arg:
        print("Invalid arguments.")
        sys.exit(1)
    
    # Generate game and run for specified number of iterations    
    generation_arg = int(generation_arg)
    game_board = pcw.unpickle_conway_board(input_arg, verbose = verbose_arg)
    game = ConwayGameOfLife(board = game_board)
    game.run_iterations(generation_arg, verbose = verbose_arg)
    pcw.pickle_conway_board(game.board, filename = output_arg, verbose = verbose_arg)

if __name__ == "__main__":
    main()