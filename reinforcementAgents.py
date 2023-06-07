
from enum import Enum
from typing import Optional
from environment import Environment
from game import GameState, AgentState, Agent, Action
from abc import ABC, abstractmethod
import random
from playerAgents import PlayerAgent
import util
from util import Vector2d
from functools import partial
from dataclasses import dataclass
import numpy as np
import math
import time
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ValueEstimationAgent(Agent):
    alpha: float = 0.002
    epsilon: float = 0.05
    gamma: float = 0.8
    lambd: float = 0.9

    @abstractmethod
    def getQValue(self, state, action):
        raise NotImplementedError

    @abstractmethod
    def getValue(self, state):
        raise NotImplementedError

    @abstractmethod
    def getPolicy(self, state):
        raise NotImplementedError

    @abstractmethod
    def getAction(self, state):
        raise NotImplementedError


class FeatureExtractor(ABC):

    feature_names: list[str]

    @abstractmethod
    def getFeatures(self, S: GameState, A: Action) -> dict:
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        raise NotImplementedError


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, S, A: Action) -> dict:
        feats = util.Counter()
        feats[(S, A)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, S, A: Action) -> dict:
        feats = util.Counter()
        feats[S] = 1.0
        feats[f'x={S[0]}'] = 1.0
        feats[f'y={S[0]}'] = 1.0
        feats[f'action={A}'] = 1.0
        return feats


class BetterExtractor(FeatureExtractor):
    def getFeatures(self, S, A: Action) -> dict:
        # TODO
        pass


class ReinforcementAgent(ValueEstimationAgent):

    def getPolicy(self, S: GameState) -> Action:
        legal = S.getLegalActions()
        random.shuffle(legal)
        return max(legal, key=partial(self.getQValue, S), default=None)

    def getValue(self, S: GameState) -> float:
        return max(map(partial(self.getQValue, S), S.getLegalActions()), default=0.0)

    @abstractmethod
    def getQValue(self, S: GameState, A: Action):
        raise NotImplementedError

    def getAction(self, S: GameState) -> Action:
        return self.getPolicy(S)

    def getTrainAction(self, S: GameState) -> Action:
        return random.choice(S.getLegalActions()) if util.flipCoin(self.epsilon) else self.getPolicy(S)


class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()

    def getQValue(self, S: GameState, A: Action):
        return self.values[(S, A)]

    def update(self, S, A, S_, R, done: bool):
        self.values[(S, A)] = (1 - self.alpha) * self.values[(S, A)] + \
            self.alpha * (R + self.gamma * (0 if done else self.getValue(S_)))

    def train(self, env: Environment):
        env.resetState()
        S = env.getCurrentState()
        while True:
            A = self.getTrainAction(S)
            S_, R, done = env.takeAction(A)
            self.update(S, A, S_, R, done)
            S = S_
            if done:
                break



class SarsaAgent(ReinforcementAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()

    def getQValue(self, S, A: Action):
        return self.values[(S, A)]

    def update(self, S, A, R, S_, A_, done: bool):
        self.values[(S, A)] = (1 - self.alpha) * self.values[(S, A)] + \
            self.alpha * (R + self.gamma * (0 if done else self.getQValue(S_, A_)))

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(
                self.lastState, self.lastAction, state, reward)
        return state

    def train(self, env: Environment):
        env.resetState()
        S = env.getCurrentState()
        A = self.getTrainAction(S)
        while True:
            S_, R, done = env.takeAction(A)
            A_ = self.getTrainAction(S_) if not done else None
            self.update(S, A, R, S_, A_, done)
            S = S_
            A = A_
            if done:
                break


class SarsaLambdaAgent(SarsaAgent):

    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()
        self.eligibility = util.Counter()

    def update(self, S, A, R, S_, A_, done: bool):
        delta = R + self.gamma * (0 if done else self.getQValue(S_, A_)) - self.getQValue(S, A)
        self.eligibility[(S, A)] += 1
        for (S, A), e in list(self.eligibility.items()):
            self.values[(S, A)] += self.alpha * delta * e
            self.eligibility[(S, A)] *= self.gamma * self.lambd

    def getQValue(self, S, A: Action) -> float:
        return self.values[(S, A)]

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(
                self.lastState, self.lastAction, state, reward)
        return state


class PlayerQAgent(QLearningAgent, PlayerAgent):
    def __init__(self, **args):
        """
        alpha    - learning rate
        epsilon  - exploration rate
        """
        PlayerAgent.__init__(self)
        QLearningAgent.__init__(self, **args)

    def getAction(self, S):
        # input()
        A = QLearningAgent.getAction(self, S)
        return A


class MyFeatures(FeatureExtractor):

    feature_names = [
        "aliveNum",
        "deadNum",
        "d2WallUp",
        "d2WallDown",
        "d2WallLeft",
        "d2WallRight",
        "minD2Wall",
        "closestAliveGhostAΔx",
        "closestAliveGhostAΔy",
        "closestAliveGhostBΔx",
        "closestAliveGhostBΔy",
        "ghostsSameX",
        "ghostsSameY",
        "playerGhostsSameX",
        "playerGhostsSameY",
        "closestSameXGhostPairΔx",
        "closestSameYGhostPairΔy",
        "closestDeadGhostΔx",
        "closestDeadGhostΔy"
    ]
    
    def getFeatures(self, s: GameState, a: Action) -> dict:
        # if a is None:
            # return {}
        # s = s.getNextState(a)
        playerPos = s.getPlayerPosition()
        sortedAliveGhostPositions = sorted(s.getAliveGhostPositions(), key=lambda ghostPos: Vector2d.manhattanDistance(ghostPos, playerPos))
        sortedDeadGhostPositions = sorted(s.getDeadGhostPositions(), key=lambda ghostPos: Vector2d.manhattanDistance(ghostPos, playerPos))
        features = {
            "aliveNum": len(sortedAliveGhostPositions),
            "deadNum": len(sortedDeadGhostPositions),
            "d2WallUp": playerPos.y - 1,
            "d2WallDown": s.layout.height - playerPos.y,
            "d2WallLeft": playerPos.x - 1,
            "d2WallRight": s.layout.width - playerPos.x,
        }
        features["minD2Wall"] = min([features["d2WallUp"], features["d2WallDown"], features["d2WallLeft"], features["d2WallRight"]])
        if len(sortedAliveGhostPositions) >= 2:
            features["closestAliveGhostAΔx"] = playerPos.x - sortedAliveGhostPositions[0].x
            # features["1/closestAliveGhostAΔx"] = 1 / (abs(playerPos.x - sortedAliveGhostPositions[0].x) + 1)
            features["closestAliveGhostAΔy"] = playerPos.y - sortedAliveGhostPositions[0].y
            # features["1/closestAliveGhostAΔy"] = 1 / (abs(playerPos.y - sortedAliveGhostPositions[0].y) + 1)
            features["closestAliveGhostBΔx"] = playerPos.x - sortedAliveGhostPositions[1].x
            # features["1/closestAliveGhostBΔx"] = 1 / (abs(playerPos.x - sortedAliveGhostPositions[1].x) + 1)
            features["closestAliveGhostBΔy"] = playerPos.y - sortedAliveGhostPositions[1].y
            # features["1/closestAliveGhostBΔy"] = 1 / (abs(playerPos.y - sortedAliveGhostPositions[1].y) + 1)
            ghostsOnX = [0 for _ in range(s.layout.width + 1)]
            ghostsOnY = [0 for _ in range(s.layout.height + 1)]
            for ghostPos in sortedAliveGhostPositions:
                ghostsOnX[ghostPos.x] += 1
                ghostsOnY[ghostPos.y] += 1
            ghostsOnX = [0 if i < 2 else 1 for i in ghostsOnX]
            ghostsOnY = [0 if i < 2 else 1 for i in ghostsOnY]
            features["ghostsSameX"] = sum(ghostsOnX)
            features["ghostsSameY"] = sum(ghostsOnY)
            features["playerGhostsSameX"] = int(ghostsOnX[playerPos.x] >= 1)
            features["playerGhostsSameY"] = int(ghostsOnY[playerPos.y] >= 1)
            if features["ghostsSameX"] > features["playerGhostsSameX"]:
                features["closestSameXGhostPairΔx"] = min([x - playerPos.x if x!=playerPos.x and n!=0 else 100 for x,n in enumerate(ghostsOnX)], key=abs)
            if features["ghostsSameY"] > features["playerGhostsSameY"]:
                features["closestSameYGhostPairΔy"] = min([y - playerPos.y if y!=playerPos.y and n!=0 else 100 for y,n in enumerate(ghostsOnY)], key=abs)
        if len(sortedDeadGhostPositions) >= 1:
            features["closestDeadGhostΔx"] = playerPos.x - sortedDeadGhostPositions[0].x
            # features["1/closestDeadGhostΔx"] = 1 / (abs(playerPos.x - sortedDeadGhostPositions[0].x) + 1)
            features["closestDeadGhostΔy"] = playerPos.y - sortedDeadGhostPositions[0].y
            # features["1/closestDeadGhostΔy"] = 1 / (abs(playerPos.y - sortedDeadGhostPositions[0].y) + 1)
        return features

class ApproximateQAgent(PlayerQAgent):

    update_counter: int = 0
    writer = SummaryWriter('runs/test')

    def __init__(self, extractor=IdentityExtractor, **args):
        self.featExtractor = extractor
        self.weights = util.Counter()
        PlayerQAgent.__init__(self, **args)

    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights.update(weights)

    def getQValue(self, S, A: Action) -> float:
        return self.getWeights() * self.featExtractor.getFeatures(S, A)

    def update(self, S, A: Action, S_, R: float, done: bool):
        if not done:
            difference = (R + self.gamma * (0 if done else self.getValue(S_))) - self.getQValue(S, A)
            for f, v in self.featExtractor.getFeatures(S_, A).items():
                self.weights[f] += self.alpha * difference * v
                self.writer.add_scalar(f, self.weights[f], self.update_counter)
        self.update_counter += 1

class MCTSNode:
    state: GameState
    agent: Agent
    visits: int
    total_reward: float
    children: list["MCTSNode"]
    parent: Optional["MCTSNode"]

    def __init__(self, state: GameState, agent: Agent, parent: Optional["MCTSNode"] = None):
        self.state = state
        self.agent = agent
        self.parent = parent
        self.visits = 0
        self.children = []
        self.total_reward = 0

    def ucb_score(self, exploration_constant: float) -> float:
        if self.visits == 0:
            return math.inf
        exploitation = self.total_reward / self.visits
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration_constant * exploration

    def bestChild(self, exploration_constant: float) -> "MCTSNode":
        """
        Select the child with the highest UCB score.
        """
        return max(self.children, key=lambda child: child.ucb_score(exploration_constant))

    def randomChild(self) -> "MCTSNode":
        """
        Select a random child of this node.
        """
        return random.choice(self.children)

    def expand(self) -> None:
        """
        Expands the current node by creating all possible child nodes.
        """
        for a in self.state.getLegalActions(self.agent.index):
            self.children.append(
                MCTSNode(state=self.state.getNextState(a), agent=self.agent, parent=self))

    @property
    def is_terminal(self) -> bool:
        return self.state.isWin() or self.state.isLose()

    @property
    def has_children(self) -> bool:
        return len(self.children) > 0

    def backpropagate(self, reward: float) -> None:
        """
        Propagate the reward backwards and update the visit count of all ancestors.
        """
        node = self
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


class MCTSAgent(Agent):

    exploration_constant: float
    num_simulations: int

    def __init__(self, index: int = 0, exploration_constant: float = 1, num_simulations: int = 10000):
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        super().__init__(index)

    def getAction(self, state) -> Action:
        if not hasattr(self, "root"):
            self.root = MCTSNode(state, agent=self)
        childStates = [child.state for child in self.root.children]
        if state in childStates:
            self.root = self.root.children[childStates.index(state)]
            self.root.parent = None
        else:
            self.root = MCTSNode(state, agent=self)

        for _ in track(range(self.num_simulations), description="MCTS simulations", total=self.num_simulations) if not self.quiet else range(self.num_simulations):
            node = self.root

            # Selection
            while not node.is_terminal and node.has_children:
                node = node.bestChild(self.exploration_constant)

            # Expansion
            if not node.is_terminal and node.visits > 0:
                node.expand()
                if node.has_children:
                    node = node.randomChild()

            # Simulation
            while not node.is_terminal:
                a = random.choice(node.state.getLegalActions())
                node = MCTSNode(state=node.state.getNextState(
                    a), agent=self, parent=node)

            reward = node.state.getScore()

            # Backpropagation
            node.backpropagate(reward)

        random.shuffle(self.root.children)
        best_child: MCTSNode = max(self.root.children, key=lambda child: child.visits)
        self.root = best_child
        if not self.quiet:
            print(f"Best child visited {best_child.visits} times")
        return Action.from_vector(best_child.state.getAgentState(self.index).configuration.direction)

