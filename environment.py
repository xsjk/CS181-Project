from abc import ABC, abstractmethod
import random
from agentRules import Action

class Environment(ABC):

    @abstractmethod
    def getCurrentState(self) -> "GameState":
        raise NotImplementedError

    @abstractmethod
    def getLegalActions() -> list[Action]:
        raise NotImplementedError

    @abstractmethod
    def takeAction(self, action: Action) -> tuple["GameState", float]:
        """
          Performs the given action in the current
          environment state and updates the enviornment.

          Returns a (reward, nextState) pair
        """
        raise NotImplementedError

    @abstractmethod
    def resetState(self):
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
    
class PlayerEnvironment(Environment):
    
    def __init__(self, player: "Agent", startState: "GameState"):
        self.player = player
        self.startState = startState.deepCopy()
        self.state = startState

    def getCurrentState(self) -> "GameState":
        return self.state
    
    def getLegalActions(self) -> list[Action]:
        return self.state.getLegalActions(self.player.index)
    
    def takeAction(self, action: Action) -> tuple["GameState", float]:
        raise NotImplementedError
    
    def resetState(self):
        self.state = self.startState.deepCopy()


class NullRewardEnvironment(PlayerEnvironment):
    
    def takeAction(self, action: Action) -> tuple["GameState", float]:
        self.state.changeToNextState(action)
        reward: float = 0
        return (self.state, reward, self.state.isWin() or self.state.isLose())
    
class NaiveRewardEnvironment(PlayerEnvironment):

    def takeAction(self, action: Action) -> tuple["GameState", float]:
        lastScore: float = self.state.getScore()
        self.state.changeToNextState(action)
        reward: float = self.state.getScore() - lastScore
        reward /= 500
        return (self.state, reward, self.state.isWin() or self.state.isLose())

class DetectRewardEnvironment(NaiveRewardEnvironment):
    
    def takeAction(self, action: Action) -> tuple["GameState", float]:
        self.state.changeToNextState(action)
        reward = self.state.getDetectReward()
        return (self.state, reward, self.state.isWin() or self.state.isLose())


class BFSRewardEnvironment(NaiveRewardEnvironment):
    
    def takeAction(self, action: Action) -> tuple["GameState", float]:
        self.state.changeToNextState(action)
        reward = self.state.getBFSReward()
        return (self.state, reward, self.state.isWin() or self.state.isLose())


class PatternRewardEnvironment(NaiveRewardEnvironment):
    
    def takeAction(self, action: Action) -> tuple["GameState", float]:
        self.state.changeToNextState(action)
        reward = self.state.getPatternReward()
        return (self.state, reward, self.state.isWin() or self.state.isLose())





