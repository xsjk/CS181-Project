from game import Agent, GameState, Action, Direction, Actions
from itertools import product
import random
from abc import ABC, abstractmethod
from util import Vector2d

class GhostAgent(Agent):
    def __init__(self, index: int):
        assert index > 0
        super().__init__(index)

    def getAction(self, state: GameState) -> Action:
        pass

class GreedyGhostAgent(GhostAgent):
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
        # print("Ghost action is:", action)
        return action


class GhostAgentSlightlyRandom(GreedyGhostAgent):
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


class ChildGhostAgent(GhostAgent):
    index: int
    father: "GhostsAgentBase"

    def __init__(self, father: "GhostsAgentBase", index: int):
        self.index = index
        self.father = father

    def getAction(self, state: GameState) -> Action:
        return self.father.getAction(state, self.index)
    
    def __str__(self):
        class_name = type(self.father).__name__
        return f"{class_name}({self.index})"


class GhostsAgentBase(list):
    nextActions: dict[int, Action] = {}

    def __init__(self, numGhosts: int):
        super().__init__(ChildGhostAgent(self, i + 1) for i in range(numGhosts))

    def getAction(self, state: GameState, childIndex: int):
        if self.nextActions == {}:
            self.nextActions = dict(enumerate(self.getActions(state), 1))
        return self.nextActions.pop(childIndex)

    @abstractmethod
    def getActions(self, state: GameState) -> list[Action]:
        raise NotImplementedError


class GhostsAgentSample(GhostsAgentBase):
    def __init__(self, numGhosts: int):
        super().__init__(numGhosts)

    def getActions(self, state: GameState) -> list[Action]:
        LEGALACTIONSLIST: list[list[Action]] = [
            state.getLegalActions(i + 1) for i in range(state.getGhostNum())
        ]
        actionsList = random.sample(list(product(*LEGALACTIONSLIST)), 100)

        def evaluate(actions):
            nextState = state.getGhostsNextState(actions)
            return sum(
                ghostState.dead == False for ghostState in nextState.agentStates[1:]
            ) - (
                sum(
                    Vector2d.manhattanDistance(
                        ghostState.getPosition(), nextState.agentStates[0].getPosition()
                    )
                    for ghostState in nextState.agentStates[1:]
                )
                + 1
                + random.random() / 2
            )

        return max(actionsList, key=evaluate)

class SmartGhostsAgent(GhostsAgentBase):
    def __init__(self, numGhosts: int):
        super().__init__(numGhosts)

    @staticmethod
    def shuffle(actions: list[Action]):
        actions = actions.copy()
        random.shuffle(actions)
        return actions

    def getActions(self, state: GameState) -> list[Action]:
        currentLive = sum(ghostState.dead == False for ghostState in state.agentStates[1:])
        LEGALACTIONSLIST: list[list[Action]] = [
            state.getLegalActions(i + 1) for i in range(state.getGhostNum())
        ]
        actions = []
        def dfs(state: GameState):
            index = len(actions)
            if index == state.getGhostNum():
                return actions
            for action in sorted(LEGALACTIONSLIST[index], key=lambda action: Vector2d.manhattanDistance(Actions.actionToVector(action) + state.getGhostPosition(index + 1), state.getPlayerPosition())):
                nextState = state.getGhostNextState(action, index + 1)
                if sum(ghostState.dead == False for ghostState in nextState.agentStates[1:]) == currentLive:
                    actions.append(action)
                    return dfs(nextState)
            actions.pop()

        dfs(state)
        return actions

        # def evaluate(actions):
        #     nextState = state.getGhostsNextState(actions)
        #     return sum(
        #         ghostState.dead == False for ghostState in nextState.agentStates[1:]
        #     ) - (
        #         sum(
        #             Vector2d.manhattanDistance(
        #                 ghostState.getPosition(), nextState.agentStates[0].getPosition()
        #             )
        #             for ghostState in nextState.agentStates[1:]
        #         )
        #         + 1
        #         + random.random() / 2
        #     )
        #
        # return max(actionsList, key=evaluate)
