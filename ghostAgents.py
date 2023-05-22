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
    


class ChildGhostAgent(GhostAgent):
    index: int
    father: "GhostsAgent"

    def __init__(self, father: "GhostsAgent", index: int):
        self.index = index
        self.father = father

    def getAction(self, state: GameState) -> Action:
        return self.father.getAction(state, self.index)

class GhostsAgentBase(ABC):
    nextActions: dict[int, Action] = {}
    children: list[ChildGhostAgent]
    def __init__(self, numGhosts: int):
        self.children = [ChildGhostAgent(self, i+1) for i in range(numGhosts)]
    
    def getAction(self, state: GameState, childIndex: int):
        if self.nextActions == {}:
            self.nextActions = dict(enumerate(self.getActions(state), 1))
        action = self.nextActions[childIndex]
        del self.nextActions[childIndex]
        return action

    @abstractmethod
    def getActions(self, state: GameState) -> list[Action]:
        raise NotImplementedError


class GhostsAgent(GhostsAgentBase):
    def __init__(self, numGhosts: int):
        super().__init__(numGhosts)

    def getActions(self, state: GameState) -> list[Action]:
        actions = []
        for i in range(1,state.getGhostNum()+1):
            player_pos = state.getPlayerPosition()
            pos = state.getGhostPosition(i)
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
            actions.append(action)
        return actions
        #print("Ghost action is:", action)

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
