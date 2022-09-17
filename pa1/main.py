import random as rn
import numpy as np
import enum


# Enumerate the moves for use later
class Move(enum):
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3


# Number of rows and cols
ROWS = 4
COLS = 4

# create a 4x4 grid to act as the board
grid = np.zeros((ROWS, COLS))

# state 15 is the win state as it is the only location with a non-negative reward
WIN_STATE = (3, 3)

# state 8 is the teleporter so a special TELE_STATE is needed
TELE_STATE = (0, 2)

# Flag for if deterministic or stochastic
DETERMINISTIC = True

# Since there is not a required starting space lets randomize it.
# It will not change once the program starts. It is just randomizing the first iteration.
# So there is a chance it randomly choose state 15 and kinda instantly 'wins'
START = (rn.randint(0, ROWS), rn.randint(0, COLS))


class State:
    def __init__(self, state=START):
        self.board = grid
        self.isEnd = False
        self.state = state

    # Being in state 15 is the only way to get a 0 reward.
    # Tele state as a reward of -2 if going left.
    # Else if any other state or action taken then the reward is -1
    def reward(self, action):
        if self.state == WIN_STATE:
            return 0
        elif self.state == TELE_STATE and action == Move.LEFT:
            return -2
        else:
            return -1

    # determine based on current state and what action is being taken what the next state will be
    def next_state(self, action):
        if DETERMINISTIC:
            if action == Move.UP:
                return
            elif action == Move.DOWN:
                return
            elif action == Move.LEFT:
                return
            elif action == Move.RIGHT:
                return

        return self.state

    def show_board(self):
        self.board[self.state] = 1
        for i in range(0, ROWS):
            print('-----------------')
            out = '|'
            for j in range(0, COLS):
                if self.board[i, j] == 1:
                    token = '*'
                elif self.board[i, j] == -1:
                    token = 'z'
                elif self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')

    # start making the agent/player all they need to do is being able to access the current state
    # and choose a move to take. Going to implement q-learning.

