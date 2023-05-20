from game import *

class normEnemy(Agent):
    def __init__(self, index:int):
        super(normEnemy, self).__init__(index)
        self.color = COLORS["enemy"]

    def getAction(self, state:GameState):
        """
        return a move string given player's position
        if overlap with player, return ""
        """
        player_pos = state.getPlayerPosition()
        pos = state.getEnemyPosition(self.index)
        action = ""
        if player_pos[1] > pos[1]:
            action += "S"
        elif player_pos[1] < pos[1]:
            action += "N"
        
        if player_pos[0] > pos[0]:
            action += "E"
        elif player_pos[0] < pos[0]:
            action += "W"
        print("Enemy action is:",action)
        return action