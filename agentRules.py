from collections.abc import Iterator
import pygame
from util import *
from enum import Enum
from vector import Vector2d

class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.length = min(width, height)
        self.sizeTuple = (self.width, self.height)


TITLE = "Robots"
MAP_SIZE = Size(20, 20)
TILE_SIZE = Size(30, 30)
WINDOW_SIZE = Size(MAP_SIZE.width * TILE_SIZE.width,
                   MAP_SIZE.height * TILE_SIZE.height)


def isPosValid(x: int, y: int) -> bool:
    """
    (`x`, `y`) = (1, 1) is considered to be the top-left grid
    """
    return 1 <= x <= MAP_SIZE.width and 1 <= y <= MAP_SIZE.height


class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """

    def __init__(self, index: int):
        self.index = index

    def getAction(self, state: "GameState") -> "Action":
        """
        The Agent will receive a GameState (from either {player, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()


class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos: Vector2d, direction):
        assert isinstance(pos, Vector2d)
        self.pos = pos
        self.direction = direction
        # self.direction = direction

    def getPosition(self):
        return (self.pos)

    def getDirection(self):
        return self.direction

    def __eq__(self, other):
        if other == None:
            return False
        return (self.pos == other.pos and self.direction == other.direction)

    def __hash__(self):
        x = hash(self.pos)
        y = hash(self.direction)
        return hash(x + 13 * y)

    def __str__(self):
        return f"(x,y)={self.pos}, {self.direction}"

    def getNextState(self, displacement: Vector2d):
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        if displacement == Vector2d(0, 0):
            return self
        return Configuration(self.pos + displacement, displacement)


class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, color, radius, etc).
    """

    def __init__(self, startConfiguration, isPlayer):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.color = COLOR['ghost']
        if isPlayer:
            self.color = COLOR['player']
        self.radius = 30 * 0.8 / 2
        self.isPlayer = isPlayer
        # state below potentially used for contest only
        self.numCarrying = 0
        self.numReturned = 0

    def __str__(self):
        if self.isPlayer:
            return f"Player: {self.configuration}"
        else:
            return f"Ghost: {self.configuration}"

    def __eq__(self, other):
        if other == None:
            return False
        return self.configuration == other.configuration

    def __hash__(self):
        return hash(hash(self.configuration))

    def copy(self):
        state = AgentState(self.start, self.isPlayer)
        state.configuration = self.configuration
        state.numCarrying = self.numCarrying
        state.numReturned = self.numReturned
        return state

    def getPosition(self):
        if self.configuration == None:
            return None
        return self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection()

    def getColor(self):
        return self.color

    def getRadius(self):
        return self.radius


class Direction(Enum):
    NORTH = 'N'
    SOUTH = 'S'
    EAST = 'E'
    WEST = 'W'
    NORTHWEST = 'NW'
    NORTHEAST = 'NE'
    SOUTHWEST = 'SW'
    SOUTHEAST = 'SE'

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol
    
    @property
    def vector(self) -> Vector2d:
        match self:
            case Direction.NORTH:
                return Vector2d(0, -1)
            case Direction.SOUTH:
                return Vector2d(0, 1)
            case Direction.EAST:
                return Vector2d(1, 0)
            case Direction.WEST:
                return Vector2d(-1, 0)
            case Direction.NORTHWEST:
                return Vector2d(-1, -1)
            case Direction.NORTHEAST:
                return Vector2d(1, -1)
            case Direction.SOUTHWEST:
                return Vector2d(-1, 1)
            case Direction.SOUTHEAST:
                return Vector2d(1, 1)
    
COLOR = {
    "default": pygame.colordict.THECOLORS["gray0"],
    "tileBg0": pygame.colordict.THECOLORS["gray80"],
    "tileBg1": pygame.colordict.THECOLORS["gray90"],
    "player": pygame.colordict.THECOLORS["cornflowerblue"],
    "ghost": pygame.colordict.THECOLORS["firebrick"]
}

GHOST_NUMBER = 10

class Action(Enum):
    N = Direction.NORTH
    S = Direction.SOUTH
    E = Direction.EAST
    W = Direction.WEST
    NW = Direction.NORTHWEST
    NE = Direction.NORTHEAST
    SW = Direction.SOUTHWEST
    SE = Direction.SOUTHEAST
    TP = "TP"

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    TOLERANCE = .001

    @staticmethod
    def actionToVector(action: Action, speed=1.0) -> Vector2d:
        if action == Action.TP:
            # TODO:
            pass
        else:
            direction = action.value
            return direction.vector * speed

    @staticmethod
    def getPossibleActions(config: Configuration) -> list[Action]:
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # print("The pos are",config.pos)
        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE):
            return [config.getDirection()]

        for action in Action:
            if action == Action.TP:
                # TODO:
                pass
            else:
                dir = action.value
                if isPosValid(*(dir.vector + config.pos)):
                    possible.append(action)

        return possible

    # TODO: 之后这里要改写
    @staticmethod
    def getLegalNeighbors(position, walls):
        x, y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir in Direction:
            dx, dy = dir.vector
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width:
                continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height:
                continue
            if not walls[next_x][next_y]:
                neighbors.append((next_x, next_y))
        return neighbors

    # @staticmethod
    # def getSuccessor(position, action):
    #     dx, dy = Actions.directionToVector(action)
    #     x, y = position
    #     return (x + dx, y + dy)
