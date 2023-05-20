from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import ReflexAgent, AlphaBetaAgent, ExpectimaxAgent

if __name__ == "__main__":
    runGames(ReflexAgent(), [GhostAgent(i+1) for i in range(4)])