from agentRules import AgentState, Direction
from game import Agent, GameState, Action
from playerAgents import PlayerAgent
import random
import util
from functools import partial
from vector import Vector2d

INF = float('inf')


class ReflexAgent(PlayerAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves: list[Action] = gameState.getLegalActions()

        # Choose one of the best actions
        scores: list[float] = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore: float = max(scores)
        bestIndices: list[int] = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action: str):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Player position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Player having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState: GameState = currentGameState.getPlayerNextState(action)
        newPos: Vector2d = childGameState.getPlayerPosition()
        newGhostStates: list[AgentState] = childGameState.getGhostStates()

        ghostPos: list[Vector2d] = [g.getPosition() for g in newGhostStates]
        # if not new ScaredTimes new state is ghost: return lowest value
        if newPos in ghostPos:
            return -1
        else:
            return -1 / min(map(partial(Vector2d.manhattanDistance, newPos), ghostPos))


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Player GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(PlayerAgent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPlayerAgent, AlphaBetaPlayerAgent & ExpectimaxPlayerAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Player is always agent index 0
        self.evaluationFunction = eval(evalFn)
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def getAction(self, s: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Player, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

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


class AlphaBetaAgent(MultiAgentSearchAgent):

    def getAction(self, s: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        v = -INF
        α = -INF
        β = INF
        a_ = None

        f = lambda s, d=0, i=0, α=-INF, β=INF: \
            self.evaluationFunction(s) if d == self.depth or s.isLose() or s.isWin() else \
            self.α_value(s, d, i, α, β) if i == 0 else \
            self.β_value(s, d, i, α, β) if 0 < i < s.getNumAgents() else \
            None

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
            self.β_value(s, d, i, α, β) if 0 < i < s.getNumAgents() else \
            None

    def α_value(self, s, d, i, α=-INF, β=INF):
        v = -INF
        for a in s.getLegalActions(i):
            v = max(v, self.n_value(s.getNextState(i, a), d, i + 1, α, β))
            if v > β:
                return v
            α = max(α, v)
        return v

    def β_value(self, s, d, i, α=-INF, β=INF):
        v = INF
        for a in s.getLegalActions(i):
            if i == s.getNumAgents() - 1:
                v = min(v, self.n_value(s.getNextState(i, a), d + 1, 0, α, β))
                if v < α:
                    return v
            else:
                v = min(v, self.n_value(s.getNextState(i, a), d, i + 1, α, β))
                if v < α:
                    return v
            β = min(β, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):

    def getAction(self, s: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return max(((self.expectScore(s.getNextState(0, a)), a) for a in s.getLegalActions()))[1]
        util.raiseNotDefined()

    def expectScore(self, s: GameState, i: int = 1, d: int = 0):
        if d == self.depth or s.isWin() or s.isLose():
            return self.evaluationFunction(s)
        A = s.getLegalActions(i)
        i_ = (i + 1) % s.getNumAgents()
        d_ = d if i_ > 0 else d + 1
        sc = (self.expectScore(s.getNextState(i, a), i_, d_) for a in A)
        if i == 0:
            return max(sc)
        else:
            return sum(sc) / len(A)
