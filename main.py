from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import ReflexAgent, AlphaBetaAgent, ExpectimaxAgent
from displayModes import PygameGraphics, NullGraphics

if __name__ == "__main__":
    runGames(
        PygameGraphics(20, 30),
        display=NullGraphics(),
        player=KeyboardAgent(), 
        ghosts=[GhostAgent(i+1) for i in range(4)],
    )
