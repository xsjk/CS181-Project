from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent

if __name__ == "__main__":
    runGames(KeyboardAgent(), [GhostAgent(i+1) for i in range(4)])
