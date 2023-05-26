
from agentRules import Action, AgentState
from display import Display
from game import GameState
from playerAgents import PlayerAgent
from util import Uniqueton, Vector2d, Size, Queue, ThreadTerminated
import pygame
from pygame.colordict import THECOLORS
from typing import Callable, Optional


class PygameGraphics(Display, metaclass=Uniqueton):
    '''
    PygameGraphics is a class that uses Pygame to graphically display the game state.
    Note that only one PygameGraphics object should be created at a time.
    '''

    TITLE = "Robots"

    COLOR = {
        "default": THECOLORS["gray0"],
        "tileBg0": THECOLORS["gray80"],
        "tileBg1": THECOLORS["gray90"],
        "player": THECOLORS["cornflowerblue"],
        "ghost": THECOLORS["firebrick"],
        "explosion": THECOLORS["orange"]
    }

    # object attributes
    surface: pygame.Surface
    map_size: Size
    tile_size: Size
    window_size: Size
    radius: float

    event_handler: Optional[Callable[[pygame.event.Event], None]] = None

    def __init__(self, map_size: Vector2d, tile_size: Vector2d):
        # if PygameGraphics.instance is not None:
        #     raise RuntimeError("Only one PygameGraphics object should be created at a time.")
        # PygameGraphics.instance = self

        self.map_size = Size(map_size.x, map_size.y)
        self.tile_size = Size(tile_size.x, tile_size.y)
        self.window_size = Size(self.map_size.width * self.tile_size.width,
                                self.map_size.height * self.tile_size.height)
        self.radius = self.tile_size.length * 0.8 / 2

    def initialize(self, state: GameState):
        print("Game begins!")
        self.surface = pygame.display.set_mode(self.window_size.sizeTuple)
        PygameGraphics.running = True
        pygame.display.set_caption(self.TITLE)
        self.update(state)

    # agent_state
    def update(self, state: GameState):
        self.draw(state)
        pygame.display.update()
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    self.finish()
                case pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.finish()
            if self.event_handler is not None:
                self.event_handler(event)

    def draw(self, state: GameState):
        for x in range(self.map_size.width):
            for y in range(self.map_size.height):
                pygame.draw.rect(
                    pygame.display.get_surface(),
                    self.COLOR[f"tileBg{(x+y)%2}"],
                    (x * self.tile_size.width, y * self.tile_size.height,
                    self.tile_size.width, self.tile_size.height)
                )
        # draw the agents
        for state in state.agentStates:
            pygame.draw.circle(
                surface = self.surface, 
                color = self.getColor(state),
                center = self.gridToPixel(state.getPosition()), 
                radius = self.radius
            )

    def finish(self):
        pygame.quit()
        PygameGraphics.running = False

    def gridToPixel(self, pos: tuple) -> tuple:
        return (pos[0] * self.tile_size.width - self.tile_size.width // 2,
                pos[1] * self.tile_size.height - self.tile_size.height // 2)
    
    @classmethod
    def getColor(cls, agent: AgentState) -> tuple[int, int, int, int]:
        if agent.isPlayer:
            return cls.COLOR["player"]
        elif agent.dead:
            return cls.COLOR["explosion"]
        else:
            return cls.COLOR["ghost"]
        



class PygameKeyboardAgent(PlayerAgent, metaclass=Uniqueton):

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

    KEY_ACTION = {k: a for a, keys in ACTION_KEYS.items() for k in keys}

    action_queue: Queue

    def __init__(self):
        PygameGraphics.event_handler = self.action_getter
        self.action_queue = Queue()

    def __del__(self):
        PygameGraphics.event_handler = None

    def action_getter(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key in self.KEY_ACTION:
                self.action_queue.push(self.KEY_ACTION[event.key])

    def getAction(self, state: GameState) -> Action:
        legal = state.getLegalActions() + [Action.TP]
        while self.action_queue.isEmpty():
            if not PygameGraphics.running:
                raise ThreadTerminated()
        action = self.action_queue.pop()
        if action not in legal:
            print(f"Invalid action {action}")
            return self.getAction(state)
        return action
        
