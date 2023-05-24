
import torch
from torch import nn
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


class QNet(nn.Module):
    def __init__(self, map_size: Vector2d):
        super().__init__()
        self.map_size = map_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # input dim: [3, map_size.x, map_size.y]
        # output dim: [9]
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * map_size.x * map_size.y, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        ).to(self.device)

    def forward(self, x):
        y = self.model(x)
        return y


class DQNAgent(QLearningAgent):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actionList = list(Action)

        # action encoding: [9]
        # N, S, E, W, NW, NE, SW, SE, TP, STOP

        # state encoding: [map_size.x, map_size.y]
        self.model = net

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 128
        self.memory = []
        self.memory_size = 10000
        self.update_freq = 1000
        self.update_counter = 0

    def getQValue(self, S, A: Action):
        index = self.actionList.index(A)
        with torch.no_grad():
            X = torch.tensor(S.toMatrix(), dtype=torch.float32, device=self.device)
            X = X.unsqueeze(0)
            ys = self.model(X)
            ys = ys.squeeze(0)
            return ys[index].item()

    def getAction(self, S: GameState) -> Action:
        with torch.no_grad():
            X = torch.tensor(S.toMatrix(), dtype=torch.float32, device=self.device)
            X = X.unsqueeze(0)
            ys = self.model(X)
            ys = ys.squeeze(0)
            legal = S.getLegalActions()
            random.shuffle(legal)
            return max(legal, key=lambda a: ys[self.actionList.index(a)], default=None)
        

    def update(self, S, A, S_, R: float) -> None:

        self.memory.append((S, A, S_, R))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        S_batch = torch.tensor([s.toMatrix() for s, _, _, _ in batch], dtype=torch.float32, device=self.device)
        A_batch = torch.tensor([self.actionList.index(a) for _, a, _, _ in batch], dtype=torch.long, device=self.device)
        S_batch_ = torch.tensor([s_.toMatrix() for _, _, s_, _ in batch], dtype=torch.float32, device=self.device)
        R_batch = torch.tensor([r for _, _, _, r in batch], dtype=torch.float32, device=self.device)

        Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
        Q_batch_ = self.model(S_batch_).max(1)[0].detach()
        target = R_batch + self.gamma * Q_batch_

        loss = self.loss_fn(Q_batch, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class ImitationAgent(DQNAgent):
    def __init__(self, net: nn.Module, expert: Agent):
        super().__init__(net)
        self.expert = expert
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, env: Environment):
        env.resetState()
        S = env.getCurrentState()
        while True:
            X = torch.tensor(S.toMatrix(), dtype=torch.float32, device=self.device)
            X = X.unsqueeze(0)
            Q = self.model(X)
            Q = Q.squeeze(0)
            A_expert = self.expert.getAction(S)
            Q_expert = torch.tensor(A_expert.onehot[:9], dtype=torch.float32, device=self.device)
            loss = self.loss_fn(Q, Q_expert)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            A = self.getTrainAction(S)
            S_, R, done = env.takeAction(A)
            if done:
                break
            self.update(S, A, S_, R)
            S = S_
            A = self.getTrainAction(S)



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
