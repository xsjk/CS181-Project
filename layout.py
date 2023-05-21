from vector import *
import random
from typing import Optional


class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, map_size: Vector2d,
                 tile_size: Vector2d,
                 ghostNum: int,
                 player_pos: Vector2d = None, ghosts_pos: list[Vector2d] = []):
        if isinstance(map_size, tuple):
            map_size = Vector2d(*map_size)
        if isinstance(tile_size, tuple):
            tile_size = Vector2d(*tile_size)
        self.map_size = map_size
        self.tile_size = tile_size
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

        if (player_pos in ghosts_pos):
            raise Exception("Player and ghost have the same initial position!")

        for i in range(len(ghosts_pos), self.ghostNum):
            pos = player_pos
            while pos in self.agentPositions:
                pos = Vector2d(random.randint(1, self.map_size.x),
                               random.randint(1, self.map_size.y))
            ghosts_pos.append(pos)

        self.agentPositions += ghosts_pos[0:self.ghostNum]
        print(self.agentPositions)

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