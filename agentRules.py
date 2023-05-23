from collections.abc import Iterator
import pygame
from util import *
from layout import Layout
from enum import Enum
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, index: int):
        self.index = index

    @abstractmethod
    def getAction(self, state) -> "Action":
        """
        The Agent will receive a GameState (from either {player, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raise NotImplementedError


class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.

    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, pos: Vector2d, direction: Vector2d):
        assert isinstance(pos, Vector2d)
        assert isinstance(direction, Vector2d)
        self.pos = pos
        self.direction = direction
        # self.direction = direction

    def getPosition(self) -> Vector2d:
        return self.pos

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
    dead: bool

    def __init__(self, startConfiguration, isPlayer):
        self.dead = False
        self.start = startConfiguration
        self.configuration: Configuration = startConfiguration
        self.color = COLOR['ghost']
        if isPlayer:
            self.color = COLOR['player']
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
        return hash(self.configuration)

    def copy(self):
        state = AgentState(self.start, self.isPlayer)
        state.color = self.color
        state.dead = self.dead
        state.configuration = self.configuration
        state.numCarrying = self.numCarrying
        state.numReturned = self.numReturned
        return state

    def getPosition(self) -> Vector2d:
        if self.configuration == None:
            return None
        return self.configuration.getPosition()

    def getDirection(self):
        return self.configuration.getDirection()

    def getColor(self):
        return self.color


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
            
    @staticmethod
    def random() -> "Direction":
        return random.choice(list(Direction))
    
COLOR = {
    "default": pygame.colordict.THECOLORS["gray0"],
    "tileBg0": pygame.colordict.THECOLORS["gray80"],
    "tileBg1": pygame.colordict.THECOLORS["gray90"],
    "player": pygame.colordict.THECOLORS["cornflowerblue"],
    "ghost": pygame.colordict.THECOLORS["firebrick"],
    "explosion": pygame.colordict.THECOLORS["orange"]
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
    STOP = "STOP"

    @property
    def vector(self) -> Vector2d:
        match self:
            case Action.TP:
                raiseNotDefined()
            case Action.STOP:
                return Vector2d(0, 0)
            case _:
                return self.value.vector
            
    @staticmethod
    def random() -> "Action":
        return random.choice(list(Action))
    
    @staticmethod
    def from_vector(vector: Vector2d) -> "Action":
        code = ""
        if vector.y > 0:
            code += "S"
        elif vector.y < 0:
            code += "N"
        if vector.x > 0:
            code += "E"
        elif vector.x < 0:
            code += "W"
        if code == "":
            code = "STOP"
        return Action[code]

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """

    @staticmethod
    def actionToVector(action: Action, speed=1.0) -> Vector2d:
        return action.vector * speed
        
    @staticmethod
    def translateVector(gameState) -> Vector2d:
        mapsize = gameState.getMapSize()
        vector = Vector2d(random.randint(-mapsize.x,mapsize.x),random.randint(-mapsize.y,mapsize.y))
        goat_pos = gameState.getPlayerPosition() + vector
        while not Actions.isPosValid(*goat_pos,*mapsize) or not Actions.isPosSafe(*goat_pos,gameState):
            #print("The goat is:",goat_pos)
            vector = Vector2d(random.randint(-mapsize.x,mapsize.x),random.randint(-mapsize.y,mapsize.y))
            goat_pos = gameState.getPlayerPosition() + vector
        return vector

    @staticmethod
    def getPossibleActions(config: Configuration,layout:Layout) -> list[Action]:
        def isValid(action: Action) -> bool:
            if action == Action.TP:
                # TODO:
                pass
            else:
                return Actions.isPosValid(*(action.vector + config.pos),layout.width,layout.height)
        return list(filter(isValid, Action))

    @staticmethod
    def isPosValid(x: int, y: int, width:int, height:int) -> bool:
        """
        (`x`, `y`) = (1, 1) is considered to be the top-left grid
        """
        return 1 <= x <= width and 1 <= y <= height

    @staticmethod
    def isPosSafe(x: int, y: int, gameState) -> bool:
        pos = Vector2d(x,y)
        ghost_positions = gameState.getGhostPositions()
        for ghost_pos in ghost_positions:
            if Vector2d.manhattanDistance(pos,ghost_pos) <= 2:
                return False
        return True

    # TODO: 之后这里要改写
    @staticmethod
    def getLegalNeighbors(position, walls):
        neighbors = []
        for dir in Direction:
            x_, y_ = dir.vector + position
            if x_ < 0 or x_ == walls.width:
                continue
            if y_ < 0 or y_ == walls.height:
                continue
            if not walls[x_][y_]:
                neighbors.append((x_, y_))
        return neighbors
