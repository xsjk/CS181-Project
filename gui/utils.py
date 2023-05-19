import pygame
from pygame import Surface


class Size:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.length = min(width, height)
        self.sizeTuple = (self.width, self.height)

class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    def get(self) -> tuple[int, int]:
        return (self.x, self.y)

TITLE = "Robots"
MAP_SIZE = Size(20, 20)
TILE_SIZE = Size(30, 30)
WINDOW_SIZE = Size(MAP_SIZE.width * TILE_SIZE.width, MAP_SIZE.height * TILE_SIZE.height)

COLORS = {
    "default": pygame.colordict.THECOLORS["gray0"],
    "tileBg0": pygame.colordict.THECOLORS["gray80"],
    "tileBg1": pygame.colordict.THECOLORS["gray90"],
    "player": pygame.colordict.THECOLORS["cornflowerblue"],
    "enemy": pygame.colordict.THECOLORS["firebrick"]
}

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
VALID_ACTIONS: set[str] = set(ACTION_KEYS.keys())

ENEMY_NUMBER = 10


def isOdd(x: int) -> bool:
    return bool(x % 2)

def isPosValid(x: int, y: int) -> bool:
    """
    (`x`, `y`) = (1, 1) is considered to be the top-left grid
    """
    return 1 <= x <= MAP_SIZE.width and 1 <= y <= MAP_SIZE.height

def gridToPixel(pos: Position) -> tuple[int, int]:
    return (pos.x * TILE_SIZE.width - TILE_SIZE.width // 2, pos.y * TILE_SIZE.height - TILE_SIZE.height // 2)


class Robots:
    def __init__(self, surface: Surface, x: int, y: int):
        """
        `x` and `y` have unit of 1 grid
        """
        self.surface = surface
        self.pos = Position(x, y)
        self.color = COLORS["default"]
        self.radius = TILE_SIZE.length * 0.8 / 2
    def move(self, action: str) -> None:
        assert action in VALID_ACTIONS, f"move action \"{action}\" is invalid"
        deltaX, deltaY = 0, 0
        for a in action:
            match a:
                case "N":
                    deltaY -= 1
                case "S":
                    deltaY += 1
                case "W":
                    deltaX -= 1
                case "E":
                    deltaX += 1
        if isPosValid(self.pos.x + deltaX, self.pos.y + deltaY):
            self.pos.x += deltaX
            self.pos.y += deltaY
    def draw(self) -> None:
        pygame.draw.circle(self.surface, self.color, gridToPixel(self.pos), self.radius)

class Player(Robots):
    def __init__(self, surface: Surface, x: int, y: int):
        super(Player, self).__init__(surface, x, y)
        self.color = COLORS["player"]

class Enemy(Robots):
    def __init__(self, surface: Surface, x: int, y: int):
        super(Enemy, self).__init__(surface, x, y)
        self.color = COLORS["enemy"]
    def chase(self, player: Player) -> str:
        """
        return a move string given player's position
        if overlap with player, return ""
        """
        action = ""
        if player.pos.y > self.pos.y:
            action += "S"
        elif player.pos.y < self.pos.y:
            action += "N"
        if player.pos.x > self.pos.x:
            action += "E"
        elif player.pos.x < self.pos.x:
            action += "W"
        return action
