from player import *

class HumanPlayer(Player):

    def __init__(self, coin_type, exploration_factor=1):
        self.print_value = False
        self.exp_factor = exploration_factor
        self.coin_type = coin_type

        Player.__init__(self, coin_type)

    @staticmethod
    def choose_move(state, winner, learn):
        # print(state)
        idx = int(input('Choose move number: ')) - 1
        return idx

    def save_memory(self):
        pass

    #  """A class that represents a human player in the game"""
    
    # def __init__(self, coin_type):
    #     """
    #     Initialize a human player with their coin type
    #     """
    #     Player.__init__(self, coin_type)