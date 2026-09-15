"""
Minimax + Alpha-Beta Pruning for the Tic-Tac-Toe environment.

Current board encoding in this repo:
    -1 = empty
     0 = O
     1 = X

Search convention used here:
    X (1) is MAX
    O (0) is MIN
    
"""
import numpy as np
from tictactoe.tic_tac_toe import TicTacAssignment

PLAIN_NODES = 0
AB_NODES = 0

# Strong move ordering for Tic-Tac-Toe:
# center, then corners, then edges

PREFERRED_ORDER = np.array([4, 0, 2, 6, 8, 1, 3, 5, 7])

plain_cache = {}
ab_cache = {}


def reset_counters():
    global PLAIN_NODES, AB_NODES
    PLAIN_NODES = 0
    AB_NODES = 0


def clear_caches():
    plain_cache.clear()
    ab_cache.clear()


def ordered_moves(state):
   
    # Return legal moves in a pruning-friendly order.
    
    return PREFERRED_ORDER[np.isin(PREFERRED_ORDER,state.get_legal_moves())]


def board_key(state):
    
    # Hashable state key for memoization.
    
    return (tuple(int(x) for x in state.state.flatten()), int(state.current_player))



def terminal_score(state, depth):
    """
    Depth-aware terminal score.

    utility():
        +1 => X win
        -1 => O win
         0 => draw

    We scale wins/losses by depth so the agent prefers:
    - faster wins
    - slower losses
    """
    result = state.utility()

    if result == 1:
        return 10 - depth
    if result == -1:
        return depth - 10
    return 0


# Plain minimax

def minimax(state):
    """
    Return the best move for the current player using plain minimax.
    X (1) is MAX, O (0) is MIN.
    """
    best_move = None

    if state.current_player == 1:  # X = MAX
        best_value = float("-inf")
        for move in ordered_moves(state):
            move = int(move)
            child = state.make_move(move)
            value = min_value(child, depth=1)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    else:  # O = MIN
        best_value = float("inf")
        for move in ordered_moves(state):
            move = int(move)
            child = state.make_move(move)
            value = max_value(child, depth=1)

            if value < best_value:
                best_value = value
                best_move = move

        return best_move


def max_value(state, depth=0):
    global PLAIN_NODES
    PLAIN_NODES += 1

    if state.is_terminal():
        return terminal_score(state, depth)

    key = board_key(state)
    if key in plain_cache:
        return plain_cache[key]

    v = float("-inf")
    for move in ordered_moves(state):
        move = int(move)
        child = state.make_move(move)
        v = max(v, min_value(child, depth + 1))

    plain_cache[key] = v
    return v


def min_value(state, depth=0):
    global PLAIN_NODES
    PLAIN_NODES += 1

    if state.is_terminal():
        return terminal_score(state, depth)

    key = board_key(state)
    if key in plain_cache:
        return plain_cache[key]

    v = float("inf")
    for move in ordered_moves(state):
        move = int(move)
        child = state.make_move(move)
        v = min(v, max_value(child, depth + 1))

    plain_cache[key] = v
    return v


# Alpha-beta minimax

def minimax_ab(state):
    
    # Return the best move for the current player using alpha-beta pruning.
    
    alpha = float("-inf")
    beta = float("inf")
    best_move = None

    if state.current_player == 1:  # X = MAX
        best_value = float("-inf")
        for move in ordered_moves(state):
            move = int(move)
            child = state.make_move(move)
            value = min_value_ab(child, alpha, beta, depth=1)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

        return best_move

    else:  # O = MIN
        best_value = float("inf")
        for move in ordered_moves(state):
            move = int(move)
            child = state.make_move(move)
            value = max_value_ab(child, alpha, beta, depth=1)

            if value < best_value:
                best_value = value
                best_move = move

            beta = min(beta, best_value)

        return best_move


def max_value_ab(state, alpha, beta, depth=0):
    global AB_NODES
    AB_NODES += 1

    if state.is_terminal():
        return terminal_score(state, depth)

    key = board_key(state)
    if key in ab_cache:
        return ab_cache[key]

    v = float("-inf")
    for move in ordered_moves(state):
        move = int(move)
        child = state.make_move(move)
        v = max(v, min_value_ab(child, alpha, beta, depth + 1))

        if v >= beta:
            ab_cache[key] = v
            return v

        alpha = max(alpha, v)

    ab_cache[key] = v
    return v


def min_value_ab(state, alpha, beta, depth=0):
    global AB_NODES
    AB_NODES += 1

    if state.is_terminal():
        return terminal_score(state, depth)

    key = board_key(state)
    if key in ab_cache:
        return ab_cache[key]

    v = float("inf")
    for move in ordered_moves(state):
        move = int(move)
        child = state.make_move(move)
        v = min(v, max_value_ab(child, alpha, beta, depth + 1))

        if v <= alpha:
            ab_cache[key] = v
            return v

        beta = min(beta, v)

    ab_cache[key] = v
    return v


# Reporting / testing helpers

def compare_algorithms(state):
    """
    Run plain minimax and alpha-beta from the same state and
    compare node counts.
    """
    clear_caches()
    reset_counters()

    plain_move = minimax(state)
    plain_nodes = PLAIN_NODES

    clear_caches()
    reset_counters()

    ab_move = minimax_ab(state)
    ab_nodes = AB_NODES

    pruned_percent = 0.0
    if plain_nodes > 0:
        pruned_percent = ((plain_nodes - ab_nodes) / plain_nodes) * 100.0

    return {
        "plain_move": plain_move,
        "ab_move": ab_move,
        "plain_nodes": plain_nodes,
        "ab_nodes": ab_nodes,
        "pruned_percent": pruned_percent,
    }


def play_minimax_vs_minimax(state, use_alpha_beta=False):
    """
    Play a full game from the given state using minimax on both sides.
    Returns the final terminal state.
    """
    current = state.copy()

    while not current.is_terminal():
        move = minimax_ab(current) if use_alpha_beta else minimax(current)
        current = current.make_move(move)

    return current
