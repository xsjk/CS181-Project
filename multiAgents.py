from agentRules import AgentState, Direction
from game import Agent, GameState, Action
from playerAgents import PlayerAgent
from functools import partial
from util import Vector2d, type_check
from typing import Optional
import itertools

INF = float('inf')


def scoreEvaluationFunction(currentGameState: GameState, actions: list[Action] = []) -> float:
    for action in actions:
        currentGameState.changeToNextState(action)
    return currentGameState.getScore()


def inverseManhattanEvaluationFunction(currentGameState: GameState, action: Optional[Action] = None) -> float:
    # Useful information you can extract from a GameState (player.py)
    if action is not None:
        childGameState: GameState = currentGameState.getPlayerNextState(action)
        # print("Player pos now is :",currentGameState.getPlayerPosition())
    else:
        childGameState: GameState = currentGameState
    newPos: Vector2d = childGameState.getPlayerPosition()
    newGhostStates: list[AgentState] = childGameState.getGhostStates()
    ghostPos: list[Vector2d] = [g.getPosition() for g in newGhostStates]

    # if not new ScaredTimes new state is ghost: return lowest value
    if newPos in ghostPos:
        return -1
    else:
        return min(map(partial(Vector2d.manhattanDistance, newPos), ghostPos))


class TimidAgent(PlayerAgent):

    def __init__(self, evalFn=inverseManhattanEvaluationFunction):
        super().__init__()
        self.evaluationFunction = evalFn

    def getAction(self, gameState: GameState):
        legal = gameState.getLegalActions()
        return max(legal, key=partial(self.evaluationFunction, gameState))


class MultiAgentSearchAgent(PlayerAgent):
    def __init__(self, evalFn=scoreEvaluationFunction, depth: int = 3):
        super().__init__()
        self.evaluationFunction = evalFn
        self.depth = depth


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, s: GameState):
        def min_value(s, d=0, g=1): return self.evaluationFunction(s) if s.isWin() or s.isLose() or d == self.depth else \
            min((max_value(s.getNextState(g, a), d+1) if g == s.getNumAgents()-1 else
                 min_value(
                s.getNextState(g, a), d, g+1)
                for a in s.getLegalActions(g)),
                default=INF)

        def max_value(s, d=0): return self.evaluationFunction(s) if s.isWin() or s.isLose() or d == self.depth else \
            max((min_value(s.getNextState(0, a), d)
                 for a in s.getLegalActions()),
                default=-INF)
        return max(((min_value(s.getNextState(0, a)), a) for a in s.getLegalActions()))[1]


# TODO: don't know what getNextState function to use
class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, s: GameState):
        v = -INF
        α = -INF
        β = INF
        a_ = None
        # self.depth = 2
        for a in s.getLegalPlayerActions():
            s_ = s.getPlayerNextState(a)
            v_ = self.n_value(s_, 0, 1, α, β)
            if v_ > v:
                v, a_ = v_, a
            α = max(α, v)
        return a_

    def n_value(self, s, d=0, i=0, α=-INF, β=INF):
        return \
            self.evaluationFunction(s) if d == self.depth or s.isLose() or s.isWin() else \
            self.α_value(s, d, i, α, β) if i == 0 else \
            self.β_value(s, d, i, α, β) if i == 1 else \
            None

    def α_value(self, s:GameState, d, i, α=-INF, β=INF):
        v = -INF
        for a in s.getLegalActions(i):
            v = max(v, self.n_value(s.getPlayerNextState(a), d, 1, α, β))
            if v > β:
                return v
            α = max(α, v)
        return v

    def β_value(self, s:GameState, d, i, α=-INF, β=INF):
        v = INF
        # actions_list = itertools.product(*[s.getLikelyActions(i) for i in range(1,s.getGhostNum()+1)])
        actions_list = [s.getGreedyAction(i) for i in range(1,s.getGhostNum()+1)]
        # for a in actions_list:
            #print(a)
        v = min(v, self.n_value(s.getGhostsNextState(actions_list), d + 1, 0, α, β))
        if v < α:
            return v
        β = min(β, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, s: GameState) -> Action:
        # bestScore = -INF
        # for a in s.getLegalActions():
        #     nextState = s.getNextState(0, a)
        #     score = self.expectScore(nextState)
        #     if score > bestScore:
        #         bestScore, bestAction = score, a
        # return bestAction
        return max(s.getLegalActions(), key=lambda a: self.expectScore(s.getNextState(0, a)))

    def expectScore(self, s: GameState, i: int = 1, d: int = 0):
        if d == self.depth or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
        A = s.getLegalActions(i)
        i_ = (i + 1) % s.getNumAgents()
        d_ = d if i_ > 0 else d + 1
        # sc = []
        # for a in A:
        #     sc.append(self.expectScore(s.getNextState(i, a), i_, d_))
        sc = (self.expectScore(s.getNextState(i, a), i_, d_) for a in A)
        if i == 0:
            return max(sc)
        else:
            return sum(sc) / len(A)
