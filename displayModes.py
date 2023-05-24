import time
from game import GameState
from abc import ABC, abstractmethod

DRAW_EVERY = 1
DISPLAY_MOVES = False
QUIET = False  # Supresses output


class GraphicMode(ABC):
    @abstractmethod
    def initialize(self, state: GameState):
        raise NotImplementedError

    @abstractmethod
    def update(self, state: GameState):
        raise NotImplementedError

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state: GameState):
        print(state)

    @abstractmethod
    def finish(self):
        raise NotImplementedError


class NullGraphics(GraphicMode):

    def __init__(self, *args, **kwargs):
        pass

    # gamestate
    def __init__(self, *args, **kargs):
        pass
    def initialize(self, state: GameState):
        pass

    def update(self, state: GameState):
        for state in state.agentStates:
            print(state)

    def finish(self):
        pass

