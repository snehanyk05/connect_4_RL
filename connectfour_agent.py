from agent import *
from model_connectfour import *


class ConnectFourAgent(Agent):

    def __init__(self, coin_type, exploration_factor=1):

        super().__init__(coin_type, exploration_factor)
        self.model = ConnectFourModel(coin_type)
        self.prev_state = np.zeros((6, 7))

    def ava_moves(self, state, board):
        moves = np.where(state[0, :] == 0)[0]
        return moves
