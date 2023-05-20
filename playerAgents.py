from game import *
import time
from agentRules import Direction


class PlayerAgent(Agent):
    def __init__(self):
        super().__init__(0)


class KeyboardAgent(PlayerAgent):

    ACTION_KEYS = {
        Action.N: {pygame.K_UP, pygame.K_w, pygame.K_k},
        Action.S: {pygame.K_DOWN, pygame.K_s, pygame.K_j},
        Action.W: {pygame.K_LEFT, pygame.K_a, pygame.K_h},
        Action.E: {pygame.K_RIGHT, pygame.K_d, pygame.K_l},
        Action.NW: {pygame.K_q, pygame.K_y},
        Action.NE: {pygame.K_e, pygame.K_u},
        Action.SW: {pygame.K_z, pygame.K_b},
        Action.SE: {pygame.K_x, pygame.K_n},
        Action.TP: {pygame.K_SPACE}
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
                        for act in KeyboardAgent.ACTION_KEYS:
                            if event.key in KeyboardAgent.ACTION_KEYS[act]:
                                action = act
                                break
        assert action in Action, f'move action "{action}" is invalid'
        return action


class RandomAgent(PlayerAgent):
    def getAction(self, state: GameState) -> Action:
        return random.choice(state.getLegalActions(self.index))
