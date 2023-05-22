from game import *

class GhostAgent(Agent):
    def __init__(self, index: int):
        assert index > 0
        super().__init__(index)

    def getAction(self, state: GameState) -> Action:
        """
        return a move string given player's position
        if overlap with player, return ""
        """
        player_pos = state.getPlayerPosition()
        pos = state.getGhostPosition(self.index)
        dir_code = ""
        if player_pos[1] > pos[1]:
            dir_code += "S"
        elif player_pos[1] < pos[1]:
            dir_code += "N"
        if player_pos[0] > pos[0]:
            dir_code += "E"
        elif player_pos[0] < pos[0]:
            dir_code += "W"
        direction = Direction(dir_code)
        action = Action(direction)
        print("Ghost action is:", action)
        return action
    
class GhostAgentSlightlyRandom(GhostAgent):
    def __init__(self, index: int):
        assert index > 0
        super().__init__(index)

    def getAction(self, state: GameState) -> Action:
        if random.random() < 0.2:
            action = random.choice(state.getLegalActions(self.index))
            print("Ghost action is:", action)
            return action
        else:
            return super().getAction(state)
