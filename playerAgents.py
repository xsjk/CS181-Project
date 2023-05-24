from game import Agent, GameState, Action
import random

class PlayerAgent(Agent):
    def __init__(self):
        super().__init__(0)


class RandomAgent(PlayerAgent):
    def getAction(self, state: GameState) -> Action:
        return random.choice(state.getLegalActions(self.index))
