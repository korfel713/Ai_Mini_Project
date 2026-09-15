import numpy as np  # use numpy for randomness and math
from typing import Literal

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []

        self.utility = 0
        self.visits = 0

        self.untried_moves = list(state.get_legal_moves())

    # Return True if every legal move from
    # this state has a corresponding child.
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    # Return the child with the highest UCB1 score.
    # Using the formula from Section 5.4.
    def best_child(self, c=1.41, f: Literal['ucb1', 'flat']='ucb1'):
        def ucb1(child):
            if child.visits == 0:
                return float('inf')

            return (
                child.utility / child.visits +
                c * np.sqrt(np.log(self.visits) / child.visits)

            )

        if f == 'ucb1':
            return max(self.children, key=ucb1)
        if f == 'flat':
            return max(self.children, key=lambda child: child.utility / child.visits)

    # Return the move leading to the child with the most visits
    # (not the highest win rate).
    def best_move(self):
        best = max(self.children, key=lambda child: child.visits)
        return best.move

# Starting from the given node, repeatedly choose
# the best child (by UCB1) until you reach a node that is
# not fully expanded or that represents a terminal state.
def select(node):
    while not node.state.is_terminal():
        if not node.is_fully_expanded():
            return node
        else:
            node = node.best_child()

    return node

# Choose one of the untried moves,
# create a new child node for it, and return the child.
def expand(node, move=None):
    move = move if move != None else np.random.choice(node.untried_moves)
    node.untried_moves.remove(move)

    new_state = node.state.make_move(move)

    child = MCTSNode(new_state, parent=node, move=move)
    node.children.append(child)
    return child

# From the given state, play a random game to completion.
# At each step, choose a uniformly random legal move.
# Return the utility.
def simulate(state):
    current_state = state

    while not current_state.is_terminal():
        moves = current_state.get_legal_moves()
        move = np.random.choice(moves)
        current_state = current_state.make_move(move)

    return current_state.utility()

# Walk from the given node up to the root.
# Increment the visit count of every node.
# Increment the win count only for nodes where the result was
# favorable for the player who made the move.
def backpropagate(node, result):
    # while node is not None:
    #     node.visits += 1
    #     if (
    #             node.state.current_player == 1 and result == 1 or
    #             node.state.current_player == 0 and result == -1
    #     ):
    #         node.wins += 1
    #
    #     node = node.parent
    while node is not None:
        node.visits += 1
        # Count win only if the move that led here is good for that player
        if node.parent is None:
            # root node, skip
            node = node.parent
            continue

        player = node.parent.state.current_player  # who made the move
        # if (player == 1 and result == 1) or (player == 0 and result == -1):
        #     node.wins += 1
        if player == 1:
            node.utility += result
        else:
            node.utility -= result
        node = node.parent

def mcts(state, iterations=1000, root=None):
    root = root or MCTSNode(state)

    if root.visits <= iterations:
        iterations -= root.visits
    else:
        iterations = 0

    for _ in range(iterations):
        leaf = select(root)
        if not leaf.state.is_terminal():
            leaf = expand(leaf)
        result = simulate(leaf.state)
        backpropagate(leaf, result)

    return root

if __name__ == "__main__":
    from tictactoe.tic_tac_toe import TicTacAssignment

    env = TicTacAssignment()
    env.reset()


    move = mcst(env, iterations=1000)
    print("Best move:", move)