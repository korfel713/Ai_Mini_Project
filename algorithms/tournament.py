import time
import numpy as np
import matplotlib.pyplot as plt
from tictactoe.tic_tac_toe import TicTacAssignment
from algorithms.mcts import mcts, expand
from algorithms.minmax import minimax, minimax_ab

with TicTacAssignment(render_mode = 'ansi') as env:
    env.reset()
    # tree = mcts(env, 50000)
    tree = mcts(env, 100)

print('tree compiled')

def minimax_agent(state):
    return minimax(state)

def alphabeta_agent(state):
    return minimax_ab(state)

# def mcts_agent(state, iterations=1000):
#     return mcts(state, iterations=iterations)

def random_agent(state):
    return np.random.choice(state.get_legal_moves())

# def print_board(state):
#     symbols = {1: "X", 0: "O", -1: "-"}
#     board = state.state.flatten()
#     board_str = "\n".join(
#         " | ".join(symbols[int(board[i*3 + j])] for j in range(3)) for i in range(3)
#     )
#     print(board_str + "\n")

def play_game(agent_X, agent_O, verbose=False):
    state = TicTacAssignment(render_mode='ansi')
    state.reset()
    while not state.is_terminal():
        if state.current_player == 1:
            move = agent_X(state)
        else:
            move = agent_O(state)
        state = state.make_move(move)
        if verbose:
            state.render()
    return state.utility()

def play_with_mcts(agent_X, agent_O, verbose=False):
    state = TicTacAssignment(render_mode='ansi')
    state.reset()
    traversal = tree
    mcts_player = 1 if agent_X == mcts else 0 if agent_O == mcts else -1
    while not state.is_terminal():
        if state.current_player == 1 and mcts_player == 1:
            traversal = mcts(state, 5000, traversal).best_child(f='flat')
            move = traversal.move
        elif state.current_player == 0 and mcts_player == 0:
            traversal = mcts(state, 5000, traversal).best_child(f='flat')
            move = traversal.move
        elif state.current_player == 1 and mcts_player == 0:
            move = agent_X(state)
            if move in traversal.untried_moves:
                traversal = expand(traversal, move)
            else:
                traversal = traversal.children[[c.move for c in traversal.children].index(move)]
        else:
            move = agent_O(state)
            if move in traversal.untried_moves:
                traversal = expand(traversal, move)
            else:
                traversal = traversal.children[[c.move for c in traversal.children].index(move)]
        state = state.make_move(move)
        if verbose:
            state.render()
    return state.utility()


def run_tournament(num_games=100, verbose=False):
    matchups = [
        ("Minimax", minimax_agent, "Random", random_agent),
        ("mcts_1000", mcts, "Random", random_agent),
        ("Minimax", minimax_agent, "mcts_1000", mcts),
        ("mcts_1000", mcts, "Minimax", minimax_agent),
    ]
    results = {}

    for X_name, X_agent, O_name, O_agent in matchups:
        key = f"{X_name} (X) vs {O_name} (O)"
        results[key] = {"X_wins": 0, "O_wins": 0, "Draws": 0}
        for game_idx in range(num_games):
            if verbose:
                print(f"Game {game_idx+1} | {key}")
            outcome = play_with_mcts(X_agent, O_agent, verbose=verbose)
            if outcome == 1:
                results[key]["X_wins"] += 1
            elif outcome == -1:
                results[key]["O_wins"] += 1
            else:
                results[key]["Draws"] += 1
        if verbose:
            print(f"Finished matchup {key}\n")
    return results

def print_tournament_results(results, num_games):
    print(f"Results after {num_games} games per matchup:\n")
    print(f"{'Matchup':35} {'X wins':>7} {'O wins':>7} {'Draws':>7}")
    print("-"*60)
    for matchup, record in results.items():
        print(f"{matchup:35} {record['X_wins']:7} {record['O_wins']:7} {record['Draws']:7}")

def benchmark_agents_time(agents, games=5):
    """Measure average time per move for multiple agents."""
    for name, agent in agents:
        move_times = []
        for _ in range(games):
            state = TicTacAssignment()
            state.reset()
            while not state.is_terminal():
                start = time.time()
                move = agent(state)
                end = time.time()
                move_times.append(end - start)
                state = state.make_move(move)
        avg_time = np.mean(move_times)
        print(f"{name}: avg time per move = {avg_time:.4f} s")

def benchmark_mcts_iterations(iterations_list, games=5):
    """Measure average time per move for mcts at different iteration counts."""
    times = {}
    for iters in iterations_list:
        move_times = []
        for _ in range(games):
            state = TicTacAssignment()
            state.reset()
            traversal = tree
            while not state.is_terminal():
                start = time.time()
                traversal = mcts(state,iters,traversal).best_child(f='flat')
                move = traversal.move
                end = time.time()
                move_times.append(end - start)
                state = state.make_move(move)
        avg_time = np.mean(move_times)
        times[iters] = avg_time
        print(f"mcts {iters} iterations: avg {avg_time:.4f} s per move")
    return times

def plot_mcts_times(times_dict):
    iters = list(times_dict.keys())
    times = list(times_dict.values())
    plt.figure(figsize=(8,5))
    plt.plot(iters, times, marker='o')
    plt.xlabel("mcts Iterations")
    plt.ylabel("Average Time per Move (s)")
    plt.title("mcts: Iterations vs Time per Move")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    NUM_GAMES = 100

    # # Run tournament
    # results = run_tournament(num_games=NUM_GAMES, verbose=False)
    # print_tournament_results(results, NUM_GAMES)

    # # Show one game visually
    # print("\nSample Game (Minimax X vs mcts 1000 O):")
    # # play_game(minimax_agent, lambda s, tree: mcts(s, 1000, tree), verbose=True)
    # play_with_mcts(minimax_agent, mcts, verbose=True)

    # Benchmark Minimax, Alpha-Beta, Random
    agent_list = [
        ("Minimax", minimax_agent),
        ("AlphaBeta", alphabeta_agent),
        ("Random", random_agent)
    ]
    benchmark_agents_time(agent_list, games=5)

    # Benchmark mcts at different iteration counts
    iterations_list = [100, 500, 1000, 5000, 10000]
    times = benchmark_mcts_iterations(iterations_list, games=5)
    plot_mcts_times(times)