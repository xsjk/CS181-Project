import random
from util import auto_convert, type_check, Vector2d
from copy import deepcopy
from dataclasses import dataclass

@dataclass
class Layout:
    """
    A Layout manages the static information about the game board.
    """
    map_size: Vector2d
    tile_size: Vector2d
    ghost_num: int
    player_pos: Vector2d
    ghosts_pos: list[Vector2d]

    def getNumGhosts(self):
        return self.ghost_num

    def arrangeAgents(self, player_pos: Vector2d, o_ghosts_pos: list[Vector2d]):
        """
        (re)arrange the agent positions as well as generate random positions when needed
        This is useful for running multiple games

        """
        self.agentPositions = []
        ghosts_pos = deepcopy(o_ghosts_pos)
        if player_pos:
            self.agentPositions.append(player_pos)
        else:
            self.agentPositions.append(Vector2d(self.width//2, self.height//2))

        if player_pos in ghosts_pos:
            raise Exception("Player and ghost have the same initial position!")

        for _ in range(len(ghosts_pos), self.ghost_num):
            pos = player_pos
            while pos in ghosts_pos or pos == player_pos:
                pos = Vector2d(random.randint(1, self.map_size.x),
                               random.randint(1, self.map_size.y))
            ghosts_pos.append(pos)

        self.agentPositions += ghosts_pos[0:self.ghost_num]

    @property
    def width(self):
        return self.map_size.x

    @property
    def height(self):
        return self.map_size.y

    @property
    def tile_width(self):
        return self.tile_size.x

    @property
    def tile_height(self):
        return self.tile_size.y

Layout.__init__ = auto_convert(verbose=True)(Layout.__init__)