from game import runGames, trainPlayer
from ghostAgents import GhostAgent, GhostAgentSlightlyRandom, GhostsAgent, GhostsAgentSample
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent, DQNAgent, ImitationAgent
from searchAgents import MaxScoreAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout
from util import Vector2d
import pickle

if __name__ == "__main__":
    map_size = Vector2d(15, 15)
    ghosts_pos = []
    player_pos = Vector2d(7, 7)
    # playerAgent = MCTSAgent(4)
    playerAgent = pickle.load(open("ImitationAgent.pkl", "rb"))
    # ghostsAgent = GhostsAgent(4)
    ghostsAgent = [GhostAgent(i) for i in range(1,4+1)]
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghostNum = 4,
        player_pos = player_pos,
        ghosts_pos = ghosts_pos,
    )
    runGames(
        display=PygameGraphics,
        layout=layout,
        player=playerAgent,
        ghosts=ghostsAgent,
        numGames=3
    )
