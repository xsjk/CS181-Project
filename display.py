from abc import ABC, ABCMeta, abstractmethod
from asyncio import sleep

SLEEP_TIME = 0
DRAW_EVERY = 1
DISPLAY_MOVES = False
QUIET = False  # Supresses output


class Display(ABC):
    @abstractmethod
    def initialize(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(self, state):
        raise NotImplementedError

    def pause(self):
        sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    @abstractmethod
    def finish(self):
        raise NotImplementedError


class NullGraphics(Display):

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, state):
        pass

    def update(self, state):
        for state in state.agentStates:
            print(state)

    def finish(self):
        pass


if __name__ == "__main__":
    print(type(Display))
    print(type(NullGraphics))