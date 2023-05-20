import pygame
from utils import *

class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.length = min(width, height)
        self.sizeTuple = (self.width, self.height)

TITLE = "Robots"
MAP_SIZE = Size(20, 20)
TILE_SIZE = Size(30, 30)
WINDOW_SIZE = Size(MAP_SIZE.width * TILE_SIZE.width, MAP_SIZE.height * TILE_SIZE.height)

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

    def __init__(self,index = 0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
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

    def __init__(self, pos:tuple, direction):
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
        return "(x,y)="+str(self.pos)+", "+str(self.direction)

    def generateSuccessor(self, vector):
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.

        Actions are movement vectors.
        """
        x, y = self.pos
        dx, dy = vector
        direction = Actions.vectorToDirection(vector)
        return Configuration((x + dx, y+dy),direction)

class AgentState:
    """
    AgentStates hold the state of an agent (configuration, speed, color, radius, etc).
    """

    def __init__(self, startConfiguration, isPlayer):
        self.start = startConfiguration
        self.configuration = startConfiguration
        self.color = COLORS["enemy"]
        if(isPlayer): self.color = COLORS["player"]
        self.radius = 30 * 0.8 / 2
        self.isPlayer = isPlayer
        # state below potentially used for contest only
        self.numCarrying = 0
        self.numReturned = 0

    def __str__(self):
        if self.isPlayer:
            return "Player: " + str(self.configuration)
        else:
            return "Enemy: " + str(self.configuration)

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

class Directions:
    ACTION_KEYS = {
    "N": {pygame.K_UP, pygame.K_w, pygame.K_k},
    "S": {pygame.K_DOWN, pygame.K_s, pygame.K_j},
    "W": {pygame.K_LEFT, pygame.K_a, pygame.K_h},
    "E": {pygame.K_RIGHT, pygame.K_d, pygame.K_l},
    "NW": {pygame.K_q, pygame.K_y},
    "NE": {pygame.K_e, pygame.K_u},
    "SW": {pygame.K_z, pygame.K_b},
    "SE": {pygame.K_x, pygame.K_n}
    }
    
    NORTH = 'N'
    SOUTH = 'S'
    EAST = 'E'
    WEST = 'W'
    NW = 'NW'
    NE = 'NE'
    SW = 'SW'
    SE = 'SE'
    # STOP = 'T'
    
    VALID_ACTIONS: set[str] = set(ACTION_KEYS.keys())

COLORS = {
    "default": pygame.colordict.THECOLORS["gray0"],
    "tileBg0": pygame.colordict.THECOLORS["gray80"],
    "tileBg1": pygame.colordict.THECOLORS["gray90"],
    "player": pygame.colordict.THECOLORS["cornflowerblue"],
    "enemy": pygame.colordict.THECOLORS["firebrick"]
}

ENEMY_NUMBER = 10

class Actions:
    """
    A collection of static methods for manipulating move actions.
    """
    # Directions
    _directions = { Directions.WEST:  (-1, 0),
                    Directions.EAST:  (1, 0),
                    Directions.NORTH: (0, -1),
                    Directions.SOUTH: (0, 1),
                    Directions.NW: (-1, -1),
                    Directions.NE: (1, -1),
                    Directions.SW: (-1, 1),
                    Directions.SE: (1,  1) }

    _directionsAsList = [('W', (-1, 0)), ('E', (1, 0)), ('N', (0, -1)), ('S', (0, 1)),
                         ('NW',(-1,-1)),('NE',(1,-1)),('SW',(-1,1)),('SE',(1,1))]

    TOLERANCE = .001

    def reverseDirection(action):
        if action == Directions.NORTH:
            return Directions.SOUTH
        if action == Directions.SOUTH:
            return Directions.NORTH
        if action == Directions.EAST:
            return Directions.WEST
        if action == Directions.WEST:
            return Directions.EAST
        return action
    reverseDirection = staticmethod(reverseDirection)

    def vectorToDirection(vector):
        dx, dy = vector
        if dy > 0:
            if dx > 0: return Directions.NE
            return Directions.NORTH
        if dy < 0:
            if dx < 0: return Directions.SW  
            return Directions.SOUTH
        if dx < 0:
            if dy > 0: return Directions.NW
            return Directions.WEST
        if dx > 0:
            if dy < 0: return Directions.SE
            return Directions.EAST
        raiseNotDefined()
    vectorToDirection = staticmethod(vectorToDirection)

    def directionToVector(direction, speed=1.0):
        dx, dy = Actions._directions[direction]
        return (dx * speed, dy * speed)
    directionToVector = staticmethod(directionToVector)

    def getPossibleActions(config:Configuration):
        possible = []
        x, y = config.pos
        x_int, y_int = int(x + 0.5), int(y + 0.5)

        # print("The pos are",config.pos)

        # In between grid points, all agents must continue straight
        if (abs(x - x_int) + abs(y - y_int) > Actions.TOLERANCE):
            return [config.getDirection()]

        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_y = y_int + dy
            next_x = x_int + dx
            if isPosValid(next_x,next_y):
                possible.append(dir)

        return possible

    getPossibleActions = staticmethod(getPossibleActions)

    # TODO: 之后这里要改写
    def getLegalNeighbors(position, walls):
        x, y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        neighbors = []
        for dir, vec in Actions._directionsAsList:
            dx, dy = vec
            next_x = x_int + dx
            if next_x < 0 or next_x == walls.width:
                continue
            next_y = y_int + dy
            if next_y < 0 or next_y == walls.height:
                continue
            if not walls[next_x][next_y]:
                neighbors.append((next_x, next_y))
        return neighbors
    getLegalNeighbors = staticmethod(getLegalNeighbors)

    def getSuccessor(position, action):
        dx, dy = Actions.directionToVector(action)
        x, y = position
        return (x + dx, y + dy)
    getSuccessor = staticmethod(getSuccessor)

