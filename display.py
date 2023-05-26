from abc import ABC, ABCMeta, abstractmethod
from asyncio import sleep
from util import classproperty

SLEEP_TIME = 0
DRAW_EVERY = 1
DISPLAY_MOVES = False
QUIET = False  # Supresses output


class Display(ABC):

    _running: bool = False

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
    
    @property
    def running(self):
        return self._running
    
    @running.setter
    def running(self, value):
        self._running = value


class NullGraphics(Display):

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, state):
        self._running = True

    def update(self, state):
        pass
    
    def finish(self):
        self._running = False


if __name__ == "__main__":
    print(type(Display))
    print(type(NullGraphics))