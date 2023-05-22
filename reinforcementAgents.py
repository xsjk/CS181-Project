
from enum import Enum
from typing import Optional
from game import GameState, AgentState, Agent, Action
from abc import ABC, abstractmethod
import random
from playerAgents import PlayerAgent
import util
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

    @abstractmethod
    def update(self, S: GameState, A: Action, S_: GameState, R: float):
        raise NotImplementedError

    def getLegalActions(self, S: GameState) -> list[Action]:
        return self.actionFn(S)

    def observeTransition(self, S: GameState, A: Action, S_: GameState, R: float):
        self.episodeRewards += R
        self.update(S, A, S_, R)

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn=lambda state: state.getLegalActions(), **kwargs):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        super().__init__(**kwargs)

    def takeAction(self, state, action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(
                self.lastState, self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print(f'Beginning {self.numTraining} episodes of Training')

    def final(self, state):
        """
          Called by Player game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(
            self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print(
                    f'\tCompleted {self.episodesSoFar} out of {self.numTraining} training episodes')
                print(f'\tAverage Rewards over all training: {trainAvg:2f}')
            else:
                testAvg = float(self.accumTestRewards) / \
                    (self.episodesSoFar - self.numTraining)
                print(
                    f'\tCompleted {self.episodesSoFar - self.numTraining} test episodes')
                print(f'\tAverage Rewards over testing: {testAvg:.2f}')
            print(
                f'\tAverage Rewards for last {NUM_EPS_UPDATE} episodes: {windowAvg:.2f}')
            print(
                f'\tEpisode took {time.time() - self.episodeStartTime:.2f} seconds')
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print(f'{msg}\n{"-"*len(msg)}')


class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()

    def getQValue(self, S, A: Action):
        return self.values[(S, A)]

    def getAction(self, S) -> Action:
        return random.choice(self.getLegalActions(S)) if util.flipCoin(self.epsilon) else self.getPolicy(S)

    def update(self, S, A, S_, R: float) -> None:
        self.values[(S, A)] = (1 - self.alpha) * self.values[(S, A)] + \
            self.alpha * (R + self.gamma * self.getValue(S_))

    def getPolicy(self, S) -> Action:
        return max(self.getLegalActions(S), key=partial(self.getQValue, S), default=None)

    def getValue(self, S) -> float:
        return max(map(partial(self.getQValue, S), self.getLegalActions(S)), default=0.0)


class SarsaAgent(ReinforcementAgent):
    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()

    def getQValue(self, S, A: Action):
        return self.values[(S, A)]

    def getAction(self, S) -> Action:
        return random.choice(self.getLegalActions(S)) if util.flipCoin(self.epsilon) else self.getPolicy(S)

    def update(self, S, A, S_, A_, R: float) -> None:
        self.values[(S, A)] = (1 - self.alpha) * self.values[(S, A)] + \
            self.alpha * (R + self.gamma * self.getQValue(S_, A_))

    def getPolicy(self, S) -> Action:
        return max(self.getLegalActions(S), key=partial(self.getQValue, S), default=None)

    def getValue(self, S) -> float:
        return max(map(partial(self.getQValue, S), self.getLegalActions(S)), default=0.0)

    def observationFunction(self, state):
        if not self.lastState is None:
            reward = state.getScore() - self.lastState.getScore()
            self.observeTransition(
                self.lastState, self.lastAction, state, reward)
        return state


class SarsaLambdaAgent(ReinforcementAgent):

    def __init__(self, **args):
        super().__init__(**args)
        self.values = util.Counter()
        self.eligibility = util.Counter()

    def update(self, S, A, S_, A_, R: float):
        self.eligibility[(S, A)] += 1
        delta = R + self.gamma * self.getQValue(S_, A_) - self.getQValue(S, A)
        for (S, A), e in self.eligibility.items():
            self.values[(S, A)] += self.alpha * delta * e
            self.eligibility[(S, A)] *= self.gamma * self.lambd

    def getQValue(self, S, A: Action) -> float:
        return self.values[(S, A)]

    def getAction(self, S) -> Action:
        return random.choice(self.getLegalActions(S)) if util.flipCoin(self.epsilon) else self.getPolicy(S)

    def getPolicy(self, S) -> Action:
        return max(self.getLegalActions(S), key=partial(self.getQValue, S), default=None)

    def getValue(self, S) -> float:
        return max(map(partial(self.getQValue, S), self.getLegalActions(S)), default=0.0)

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
            self.children.append(MCTSNode(state=self.state.getNextState(
                self.agent.index, a), agent=self.agent, parent=self))

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

    def __init__(self, index: int = 0, exploration_constant: float = 1, num_simulations: int = 50):
        self.exploration_constant = exploration_constant
        self.num_simulations = num_simulations
        super().__init__(index)

    def getAction(self, state) -> Action:
        root = MCTSNode(state, agent=self)
        for _ in track(range(self.num_simulations), description="MCTS simulations", total=self.num_simulations):
            node = root

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
                a = random.choice(node.state.getLegalActions(self.index))
                node = MCTSNode(state=node.state.getNextState(
                    a), agent=self, parent=node)

            reward = node.state.getScore()

            # Backpropagation
            node.backpropagate(reward)

        best_child: MCTSNode = max(
            root.children, key=lambda child: child.visits)
        print(f"Best child visited {best_child.visits} times")
        return Action.from_vector(best_child.state.getAgentState(self.index).configuration.direction)
