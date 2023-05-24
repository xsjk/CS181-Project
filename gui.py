
from agentRules import Action, AgentState
from display import Display
from game import GameState
from playerAgents import PlayerAgent
from util import Vector2d, Size, isOdd
import pygame
from pygame.colordict import THECOLORS


TITLE = "Robots"

COLOR = {
    "default": THECOLORS["gray0"],
    "tileBg0": THECOLORS["gray80"],
    "tileBg1": THECOLORS["gray90"],
    "player": THECOLORS["cornflowerblue"],
    "ghost": THECOLORS["firebrick"],
    "explosion": THECOLORS["orange"]
}

class PygameGraphics(Display):
    def __init__(self, map_size: Vector2d, tile_size: Vector2d):
        self.MAP_SIZE = Size(map_size.x, map_size.y)
        self.TILE_SIZE = Size(tile_size.x, tile_size.y)
        self.WINDOW_SIZE = Size(self.MAP_SIZE.width * self.TILE_SIZE.width,
                                self.MAP_SIZE.height * self.TILE_SIZE.height)
        self.radius = self.TILE_SIZE.length * 0.8 / 2

    def initialize(self, state):
        print("Game begins!")
        self.surface: pygame.Surface = pygame.display.set_mode(self.WINDOW_SIZE.sizeTuple)
        self.update(state)

    # agent_state
    def update(self, state: GameState):
        # draw the back ground
        for x in range(self.MAP_SIZE.width):
            for y in range(self.MAP_SIZE.height):
                pygame.draw.rect(
                    pygame.display.get_surface(),
                    COLOR["tileBg0"] if isOdd(x + y) else COLOR["tileBg1"],
                    (x * self.TILE_SIZE.width, y * self.TILE_SIZE.height,
                     self.TILE_SIZE.width, self.TILE_SIZE.height)
                )

        # draw the agents
        for state in state.agentStates:
            pygame.draw.circle(
                surface = self.surface, 
                color = self.getColor(state),
                center = self.gridToPixel(state.getPosition()), 
                radius = self.radius
            )

        pygame.display.update()

    def draw(self, state):
        print(state)

    def finish(self):
        pass

    def gridToPixel(self, pos: tuple) -> Vector2d:
        return (pos[0] * self.TILE_SIZE.width - self.TILE_SIZE.width // 2,
                pos[1] * self.TILE_SIZE.height - self.TILE_SIZE.height // 2)
    
    @staticmethod
    def getColor(agent: AgentState) -> tuple[int, int, int, int]:
        if agent.isPlayer:
            return COLOR["player"]
        elif agent.dead:
            return COLOR["explosion"]
        else:
            return COLOR["ghost"]
        



class PygameKeyboardAgent(PlayerAgent):

    ACTION_KEYS = {
        Action.N: {pygame.K_UP, pygame.K_w, pygame.K_k},
        Action.S: {pygame.K_DOWN, pygame.K_s, pygame.K_j},
        Action.W: {pygame.K_LEFT, pygame.K_a, pygame.K_h},
        Action.E: {pygame.K_RIGHT, pygame.K_d, pygame.K_l},
        Action.NW: {pygame.K_q, pygame.K_y},
        Action.NE: {pygame.K_e, pygame.K_u},
        Action.SW: {pygame.K_z, pygame.K_b},
        Action.SE: {pygame.K_x, pygame.K_n},
        Action.TP: {pygame.K_t, pygame.K_SPACE}
    }

    def getAction(self, state: GameState) -> Action:
        action = None
        while action == None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    else:
                        for act in self.ACTION_KEYS:
                            if event.key in self.ACTION_KEYS[act]:
                                action = act
                                break
        assert action in Action, f'move action "{action}" is invalid'
        return action
