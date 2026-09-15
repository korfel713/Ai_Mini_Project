# AI Mini Project 3

AI Mini Project 3 is a Python project that compares search-based game-playing algorithms using Tic-Tac-Toe as the test environment.

The project was built as a team assignment at Florida Polytechnic University. This repository is my personal portfolio copy of the finished project, with the original team repository left unchanged.

## Project Overview

The program provides a Gymnasium-based Tic-Tac-Toe environment and several agents for choosing moves:

- Minimax
- Minimax with alpha-beta pruning
- Monte Carlo Tree Search (MCTS)
- Random move selection

The project also includes a tournament and benchmarking script for comparing agents and measuring average decision time at different MCTS iteration counts.

## My Contributions

My work focused primarily on Monte Carlo Tree Search and the project evaluation tools.

Based on the original commit history, I:

- Implemented the project's initial MCTS algorithm, including selection, expansion, simulation, and backpropagation
- Updated the Tic-Tac-Toe state transition behavior so search algorithms could work with copied game states
- Converted MCTS randomness and mathematical operations to NumPy
- Built the tournament script used to compare Minimax, MCTS, and random agents
- Added timing benchmarks for Minimax, alpha-beta pruning, random play, and multiple MCTS iteration counts
- Made follow-up fixes to MCTS backpropagation and tournament behavior

The final MCTS implementation also includes later refinements made by other team members.

## Project Structure

```text
algorithms/
    mcts.py         # Monte Carlo Tree Search
    minmax.py       # Minimax and alpha-beta pruning
    tournament.py   # Matchups and performance benchmarks

tictactoe/
    __init__.py
    tic_tac_toe.py  # Gymnasium environment and assignment wrapper

requirements.txt
```

## Running the Project

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the tournament and benchmark module from the repository root:

```bash
python -m algorithms.tournament
```

The tournament module contains functions for running agent-vs-agent matchups, printing results, benchmarking decision time, and plotting MCTS iteration count against average move time.

## Algorithms

### Minimax

The Minimax agent searches the Tic-Tac-Toe game tree and chooses moves assuming optimal play from both players.

The project also includes an alpha-beta version that reduces the number of nodes evaluated while preserving the same decision objective.

### Monte Carlo Tree Search

The MCTS implementation builds a search tree through four main stages:

1. Selection
2. Expansion
3. Simulation
4. Backpropagation

The search uses UCB1 during tree traversal and can reuse an existing tree between moves to reduce repeated computation.

## Environment

The Tic-Tac-Toe environment is implemented with Gymnasium and uses a 3x3 NumPy board.

Board values are:

- `-1` for an empty square
- `0` for O
- `1` for X

The assignment wrapper exposes helper methods such as `get_legal_moves()`, `make_move()`, `is_terminal()`, `check_winner()`, and `utility()` for use by the search algorithms.

## Team

The original repository includes work from:

- Nathaniel Ward
- Joseph Williams
- Sean Thompson

Original team repository:

https://github.com/nzw2cp/AI_Mini_Project_3

## Portfolio Notes

This version removes IDE-specific project files and corrects the dependency list so the repository is easier to clone and review. The algorithm and environment source is based on the team's completed project.
