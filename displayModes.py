import time
from util import *
from vector import *
from abc import ABC, abstractmethod
import pygame

DRAW_EVERY = 1
SLEEP_TIME = 0  # This can be overwritten by __init__
DISPLAY_MOVES = False
QUIET = False  # Supresses output


class GraphicMode(ABC):
    @abstractmethod
    def initialize(self, state):
        raise NotImplementedError

    @abstractmethod
    def update(self, state):
        raise NotImplementedError

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    @abstractmethod
    def finish(self):
        raise NotImplementedError


class NullGraphics(GraphicMode):
    # gamestate
    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def checkNullDisplay(self):
        return True

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.length = min(width, height)
        self.sizeTuple = (self.width, self.height)


TITLE = "Robots"

COLOR = {
    "default": pygame.colordict.THECOLORS["gray0"],
    "tileBg0": pygame.colordict.THECOLORS["gray80"],
    "tileBg1": pygame.colordict.THECOLORS["gray90"],
    "player": pygame.colordict.THECOLORS["cornflowerblue"],
    "ghost": pygame.colordict.THECOLORS["firebrick"]
}


def isOdd(x: int) -> bool:
    return bool(x % 2)


class PygameGraphics(GraphicMode):
    def __init__(self, map_size: Vector2d, tile_size: Vector2d):
        self.display = pygame.display
        self.MAP_SIZE = Size(map_size.x, map_size.y)
        self.TILE_SIZE = Size(tile_size.x, tile_size.y)
        self.WINDOW_SIZE = Size(self.MAP_SIZE.width * self.TILE_SIZE.width,
                                self.MAP_SIZE.height * self.TILE_SIZE.height)
        self.radius = self.TILE_SIZE.length * 0.8 / 2

    def initialize(self, state):
        print("Game begins!")
        self.surface = self.display.set_mode(self.WINDOW_SIZE.sizeTuple)
        self.update(state)

    # agent_state
    def update(self, state):
        # draw the back ground
        for x in range(self.MAP_SIZE.width):
            for y in range(self.MAP_SIZE.height):
                pygame.draw.rect(
                    self.display.get_surface(),
                    COLOR["tileBg0"] if isOdd(x + y) else COLOR["tileBg1"],
                    (x * self.TILE_SIZE.width, y * self.TILE_SIZE.height,
                     self.TILE_SIZE.width, self.TILE_SIZE.height)
                )

        # draw the agents

        for state in state.agentStates:
            print(state)
            pygame.draw.circle(self.surface, state.getColor(), self.gridToPixel(
                state.getPosition()), self.radius)

        self.display.update()

    def pause(self):
        time.sleep(SLEEP_TIME)

    def draw(self, state):
        print(state)

    def finish(self):
        pass

    def gridToPixel(self, pos: tuple) -> Vector2d:
        return (pos[0] * self.TILE_SIZE.width - self.TILE_SIZE.width // 2,
                pos[1] * self.TILE_SIZE.height - self.TILE_SIZE.height // 2)
