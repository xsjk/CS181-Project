
from game import GameState, Agent, Action, Actions, Direction
from multiAgents import scoreEvaluationFunction
import search
from search import SearchProblem, nullHeuristic
from functools import partial
from util import Vector2d, type_check
from typing import Callable


class PositionSearchProblem(SearchProblem):

    def __init__(self, gameState: GameState, costFn: Callable = lambda _: 1, goal: Vector2d = Vector2d(1, 1), start=None):
        super().__init__(gameState)
        self.ghosts = gameState.getWalls()
        self.startState = gameState.getPlayerPosition()
        if start != None:
            self.startState = start
        self.goal: Vector2d = goal
        self.costFn = costFn

    def getStartState(self):
        return self.startState

    def isGoalState(self, s: Vector2d) -> bool:
        isGoal = s == self.goal
        return isGoal

    def getSuccessors(self, s: Vector2d):
        successors = []
        for a in Action:
            s_ = s + a.vector
            if not self.ghosts[s.x][s.y]:
                successors.append((s_, a, self.costFn(s_)))
        return successors

    def getCostOfActions(self, actions: list[Action]):
        if actions == None:
            return 999999
        s = self.getStartState()
        cost = 0
        for action in actions:
            x, y = s + Actions.actionToVector(action)
            if self.ghosts[x][y]:
                return 999999
            cost += self.costFn((x, y))
        return cost


class LongestLiveProblem(SearchProblem):

    def getStartState(self):
        return self.startState

    def isGoalState(self, s: GameState) -> bool:
        return s.isWin() or s.isLose()

    def getSuccessors(self, s: GameState) -> list[tuple[GameState, Action, int]]:
        successors = []
        for a in s.getLegalPlayerActions():
            if a != Action.TP:
                s_ = s.getNextState(a)
                successors.append((s_, a, 1))
        return successors


class SearchAgent(Agent):
    @type_check
    def __init__(self, func: Callable = search.bfs, prob: type = PositionSearchProblem, heuristic: Callable = nullHeuristic):
        if 'heuristic' in func.__code__.co_varnames:
            func = partial(func, heuristic=heuristic)
        self.searchFunction: Callable = func

    def prepareActions(self, state: GameState):
        self.actions: list[Action] = self.searchFunction(
            self.searchType(state))
        self.actionIndex: int = 0

    def getAction(self, state: GameState) -> Action:
        if 'actionIndex' not in dir(self):
            self.prepareActions(state)
        if self.actionIndex < len(self.actions):
            action = self.actions[self.actionIndex]
            self.actionIndex += 1
            return action
        else:
            del self.actionIndex
            return self.getAction(state)


class LongestLiveAgent(SearchAgent):

    def prepareActions(self, state: GameState):
        self.actions: list[Action] = []
        self.actionIndex: int = 0
        for i, actions in enumerate(search.breadthFirstSearchIterator(LongestLiveProblem(state))):
            if len(actions) > len(self.actions):
                self.actions = actions
            print(f"searched {i} states", end='\r')
        print(f"searched {i} states")
        self.actions.append(Action.TP)


class MaxScoreAgent(SearchAgent):

    def prepareActions(self, state: GameState):
        self.actions: list[Action] = []
        self.actionIndex: int = 0
        max_score = -float('inf')
        for i, actions in enumerate(search.uniformCostSearchIterator(LongestLiveProblem(state), depth=3)):
            score = scoreEvaluationFunction(state, actions)
            if score > max_score:
                max_score = score
                self.actions = actions
            print(f"searched {i} states", end='\r')