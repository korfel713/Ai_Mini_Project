import gymnasium as gym
from gymnasium import spaces

import numpy as np

from typing import Literal

DEFAULT_REWARD_DICT = {
    'playing':  0,
    'win'    :  1,
    'draw'   : -1,
    'loss'   : -1,
    'invalid': -1,
}

PLAYER_MAPPING = {
    -1: ' ', 
    0 : 'O', 
    1 : 'X',
}

class TicTacToe(gym.Env):	
    metadata = {"render_modes": ["ansi"], "render_fps": 30}
    reward_range = (-np.inf, np.inf)
    spec = None

    action_space = spaces.Tuple([
        spaces.Discrete(9),
        spaces.Discrete(2)
    ])
    observation_space = spaces.Box(
        low=-1,
        high=1,
        shape=(3, 3), 
        dtype=np.int8
    )

    def __init__(self, **kwargs):                
        # define render_mode if your environment supports rendering
        self.render_mode = kwargs.pop('render_mode', None)
        self.reward_dict = kwargs.pop('reward_dict', DEFAULT_REWARD_DICT)

        self.info = {'end_reason': None}

    def check_win(self, player: int) -> bool:
        mask = self.state == player
        return np.array([
            (mask).all(axis=0).any(),         # horizontal
            (mask).all(axis=1).any(),         # vertical
            (mask).diagonal().all(),          # diagonal
            np.fliplr(mask).diagonal().all(), # anti-diagonal
        ]).any()
    
    def check_draw(self) -> bool:
        return (self.state != -1).all()
    
    @property
    def possible_moves(self):
        return np.argwhere(self.state.flatten() == -1).flatten()
    
    def get_reward(self, player):
        if self.info['reward'] == 'win' or self.info['reward'] == 'loss':
            self.info['reward'] = 'win' if self.check_win(player) else 'loss'

        return self.reward_dict[self.info['reward']]

    def step(self, action):
        x, y   = action[0]%3, action[0]//3
        player = action[1]
        terminated, truncated = [False]*2


        # position already taken
        if self.state[y,x] != -1:
            self.info['end_reason'] = 'Position already taken.'
            self.info['reward']     = 'invalid'
            truncated               = True
        # not in the set of action space
        elif x < 0 or x > 8:
            self.info['end_reason'] = 'Action not in action space.'
            self.info['reward']     = 'invalid'
            truncated               = True

        # valid move
        else:
            self.state[y,x] = player

            if self.check_win(player):
                self.info['end_reason'] = f'{PLAYER_MAPPING[player]} Win.'
                self.info['reward']     = 'win'
                terminated              = True
            
            elif self.check_draw():
                self.info['end_reason'] = f'Cat Scratch.'
                self.info['reward']     = 'draw'
                terminated              = True

            elif self.check_win(1-player):
                self.info['end_reason'] = f'{PLAYER_MAPPING[player]} Win.'
                self.info['reward']     = 'loss'
                terminated              = True

        return (
            self.state, 
            self.reward_dict[self.info['reward']], 
            terminated, 
            truncated, 
            self.info
        )

    def reset(self, seed: int=None, options=None):
        super().reset(seed=seed, options=options)

        self.state = self.observation_space.low.copy()
        self.info  = {
            'end_reason': '', 
            'reward': 'playing'
        }

        return self.state, self.info

    def render(self):
        board = self.state if hasattr(self, "state") else self.observation_space.low
        grid  = ''

        if self.render_mode == 'ansi':
            rows = []
            for row in board:
                symbols = [PLAYER_MAPPING.get(int(cell), "?") for cell in row]
                rows.append(f" {symbols[0]} | {symbols[1]} | {symbols[2]} ")

            grid = ("\n---+---+---\n").join(rows)
            print(grid)

        return grid
    
    def close(self):
        pass



class TicTacAssignment(TicTacToe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.current_player = 1

    def get_legal_moves(self):
        return self.possible_moves
    #
    # def make_move(self, move):
    #     # Place the current player's mark at the
    #     # given index. Return a NEW TicTacToe
    #     # object; do not modify self.
    #     self.current_player = 1 - self.current_player
    #     return self.step((move, self.current_player))


    def copy(self):
        new_state = TicTacAssignment(
            render_mode=self.render_mode,
            reward_dict=self.reward_dict.copy()
        )
        new_state.state = self.state.copy()
        new_state.info = dict(self.info)
        new_state.current_player = self.current_player
        return new_state

    def make_move(self, move):
        new_env = self.copy()
        new_env.current_player = 1-self.current_player
        new_env.step((move, self.current_player))
        return new_env
        
    def is_terminal(self):
        # Return True if the game is over, either
        # because someone has won or because all
        # squares are filled (a draw).
        return self.check_draw() or self.check_win(0) or self.check_win(1)

    def check_winner(self):
        # Return 1 if X has three in a row, -1 if
        # O has three in a row, or 0 otherwise.
        if self.check_win(1):
            return 1
        if self.check_win(0):
            return -1

        return 0

    def utility(self):
        # Return +1 if X has won, -1 if O has won,
        # or 0 for a draw. Only valid when
        # is_terminal() returns True.
        if not self.is_terminal():
            raise RuntimeWarning('Environment is not terminal.')
        
        return self.check_winner()

    def display(self):
        self.render()

    def reset(self, seed=None, options=None):
        self.current_player = 1

        return super().reset(seed, options)
