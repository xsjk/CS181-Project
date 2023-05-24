
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


@dataclass
class ValueEstimationAgent(Agent):
    alpha: float = 1.0
    epsilon: float = 0.05
    gamma: float = 0.8
    lambd: float = 0.9
    numTraining: int = 10

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
    @abstractmethod
    def getFeatures(self, S, A: Action) -> dict:
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

    def update(self, S, A, S_, R: float) -> None:
        self.values[(S, A)] = (1 - self.alpha) * self.values[(S, A)] + \
            self.alpha * (R + self.gamma * self.getValue(S_))

    def train(self, env: Environment):
        env.resetState()
        S = env.getCurrentState()
        while True:
            A = self.getTrainAction(S)
            S_, R, done = env.takeAction(A)
            self.update(S, A, S_, R)
            S = S_
            if done:
                break



class SarsaAgent(ReinforcementAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()

    def getQValue(self, S, A: Action):
        return self.values[(S, A)]

    def update(self, S, A, R, S_, A_):
        self.values[(S, A)] = (1 - self.alpha) * self.values[(S, A)] + \
            self.alpha * (R + self.gamma * self.getQValue(S_, A_))

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
            if done:
                break
            A_ = self.getTrainAction(S_)
            self.update(S, A, R, S_, A_)
            S = S_
            A = A_


class SarsaLambdaAgent(SarsaAgent):

    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()
        self.eligibility = util.Counter()
        
    def update(self, S, A, R, S_, A_) -> None:
        self.eligibility[(S, A)] += 1
        delta = R + self.gamma * self.getQValue(S_, A_) - self.getQValue(S, A)
        for (S, A), e in self.eligibility.copy().items():
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
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        PlayerAgent.__init__(self)
        QLearningAgent.__init__(self, **args)

    def getAction(self, S):
        A = QLearningAgent.getAction(self, S)
        self.takeAction(S, A)
        return A


class ApproximateQAgent(PlayerQAgent):

    def __init__(self, extractor=IdentityExtractor, **args):
        self.featExtractor = extractor
        self.weights = util.Counter()
        PlayerQAgent.__init__(self, **args)

    def getWeights(self):
        return self.weights

    def getQValue(self, S, A: Action) -> float:
        return self.getWeights() * self.featExtractor.getFeatures(S, A)

    def update(self, S, A: Action, S_, R: float):
        difference = (R + self.gamma *
                      self.computeValueFromQValues(S_)) - self.getQValue(S, A)
        for k, v in self.featExtractor.getFeatures(S, A).items():
            self.weights[k] += self.alpha * difference * v


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

        for _ in track(range(self.num_simulations), description="MCTS simulations", total=self.num_simulations):
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
        print(f"Best child visited {best_child.visits} times")
        return Action.from_vector(best_child.state.getAgentState(self.index).configuration.direction)


from deepLearningAgents import *
# This is only for compatibility with the old pickled models