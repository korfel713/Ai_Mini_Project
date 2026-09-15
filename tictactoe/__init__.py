from gymnasium.envs.registration import register

register(
    id='TicTacToe-v0',
    entry_point='tictactoe.tic_tac_toe:TicTacToe',
)

register(
    id='TicTacToe-vAssignment',
    entry_point='tictactoe.tic_tac_toe:TicTacAssignment',
)