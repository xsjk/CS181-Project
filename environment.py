from abc import ABC, abstractmethod
import random
from game import Action, GameState

class Environment(ABC):

    @abstractmethod
    def getCurrentState(self):
        """
        Returns the current state of enviornment
        """
        raise NotImplementedError

    @abstractmethod
    def getPossibleActions(self, state: GameState):
        """
          Returns possible actions the agent
          can take in the given state. Can
          return the empty list if we are in
          a terminal state.
        """
        raise NotImplementedError

    @abstractmethod
    def takeAction(self, action: Action):
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, nextState) pair
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
          Resets the current state to the start state
        """
        raise NotImplementedError
    
    def isTerminal(self):
        """
          Has the enviornment entered a terminal
          state? This means there are no successors
        """
        state = self.getCurrentState()
        actions = self.getPossibleActions(state)
        return len(actions) == 0
    


class GridworldEnvironment(Environment):

    def __init__(self, gridWorld):
        self.gridWorld = gridWorld
        self.reset()

    def getCurrentState(self):
        return self.state

    def getPossibleActions(self, state: GameState):
        return self.gridWorld.getPossibleActions(state)

    def takeAction(self, action):
        state = self.getCurrentState()
        (nextState, reward) = self.getRandomNextState(state, action)
        self.state = nextState
        return (nextState, reward)

    def getRandomNextState(self, state: GameState, action: Action, randObj=None):
        rand = -1.0
        if randObj is None:
            rand = random.random()
        else:
            rand = randObj.random()
        sum = 0.0
        successors = self.gridWorld.getTransitionStatesAndProbs(state, action)
        for nextState, prob in successors:
            sum += prob
            if sum > 1.0:
                raise Exception('Total transition probability more than one; sample failure.')
            if rand < sum:
                reward = self.gridWorld.getReward(state, action, nextState)
                return (nextState, reward)
        raise Exception('Total transition probability less than one; sample failure.')

    def reset(self):
        self.state = self.gridWorld.getStartState()


