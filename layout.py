from vector import *
import random

class Layout:
    """
    A Layout manages the static information about the game board.
    """

    def __init__(self, map_size:Vector2d,
                 tile_size:Vector2d,
                 ghostNum:int,
                 player_pos:Vector2d = None, ghosts_pos:list[Vector2d] = []):
        
        self.map_size = map_size
        self.tile_size = tile_size
        self.agentPositions = []
        self.ghostNum = ghostNum
        self.arrangeAgents(player_pos,ghosts_pos)
        # self.initializeVisibilityMatrix()

    def getNumGhosts(self):
        return self.ghostNum
    
    def arrangeAgents(self,player_pos:Vector2d,ghosts_pos:list[Vector2d]):
        if(player_pos):
            self.agentPositions.append(player_pos)
        else:
            self.agentPositions.append(Vector2d(self.width//2,self.height//2))
        
        if(player_pos in ghosts_pos):
            raise Exception("Player and ghost have the same initial position!")

        for i in range(len(ghosts_pos),self.ghostNum):
            pos = player_pos
            while pos in self.agentPositions:
                pos = Vector2d(random.randint(1, self.map_size.x), random.randint(1, self.map_size.y))
            ghosts_pos.append(pos)

        self.agentPositions += ghosts_pos[0:self.ghostNum]
        print(self.agentPositions)
    
    

