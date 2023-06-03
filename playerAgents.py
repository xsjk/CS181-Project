from game import Agent, GameState, Action
import random

class PlayerAgent(Agent):
    def __init__(self):
        super().__init__(0)

    def __str__(self):
        class_name = type(self).__name__
        return f"{class_name}()"


class RandomAgent(PlayerAgent):
    def getAction(self, state: GameState) -> Action:
        return random.choice(state.getLegalActions(self.index))
