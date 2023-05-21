from vector import *
import random
from typing import Optional

class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, width: int, height: int,
                 tile_width: int, tile_height: int,
                 ghostNum: int,
                 player_pos: Optional[Vector2d] = None, ghosts_pos: list[Vector2d] = []):

        self.width = width
        self.height = height
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.agentPositions = []
        self.ghostNum = ghostNum
        if isinstance(player_pos, tuple):
            player_pos = Vector2d(*player_pos)
        for i in range(len(ghosts_pos)):
            if isinstance(ghosts_pos[i], tuple):
                ghosts_pos[i] = Vector2d(*ghosts_pos[i])
        self.arrangeAgents(player_pos, ghosts_pos)
        # self.initializeVisibilityMatrix()

    def getNumGhosts(self):
        return self.ghostNum

    def arrangeAgents(self, player_pos: Vector2d, ghosts_pos: list[Vector2d]):
        if (player_pos):
            self.agentPositions.append(player_pos)
        else:
            self.agentPositions.append(Vector2d(self.width//2, self.height//2))

        for i in range(len(ghosts_pos), self.ghostNum):
            pos = player_pos
            while pos in self.agentPositions:
                pos = Vector2d(random.randint(1, self.width),
                               random.randint(1, self.height))
            ghosts_pos.append(pos)

        self.agentPositions += ghosts_pos[0:self.ghostNum]
        print(self.agentPositions)
