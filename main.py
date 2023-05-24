from game import runGames, trainPlayer
from ghostAgents import GhostAgent, GhostAgentSlightlyRandom, GhostsAgent, GhostsAgentSample
from playerAgents import RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent
from searchAgents import MaxScoreAgent
from display import NullGraphics
from layout import Layout
from util import Vector2d
import pickle

import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback
    traceback.install()
if pkgutil.find_loader("torch"):
    from deepLearningAgents import DQNAgent, ImitationAgent
if pkgutil.find_loader("pygame"):
    from gui import PygameKeyboardAgent
    from gui import PygameGraphics
if pkgutil.find_loader("textual"):
    from tui import TextualKeyboardAgent
    from tui import TextualGraphics


if __name__ == "__main__":
    map_size = Vector2d(15, 15)
    ghost_num = 4
    ghosts_pos = []
    player_pos = Vector2d(7, 7)
    # playerAgent = PygameKeyboardAgent()
    playerAgent = MaxScoreAgent()
    # playerAgent = pickle.load(open("ImitationAgent.pkl", "rb"))
    # ghostsAgent = GhostsAgent(4)
    ghostsAgent = [GhostAgent(i) for i in range(1, ghost_num+1)]
    # ghostsAgent = list(map(GhostAgent, range(1, ghost_num+1)))
    layout = Layout(
        map_size=map_size,
        tile_size=(30, 30),
        ghost_num=ghost_num,
        player_pos=player_pos,
        ghosts_pos=ghosts_pos,
    )
    runGames(
        display=NullGraphics,
        layout=layout,
        player=playerAgent,
        ghosts=ghostsAgent,
        # numGames=100
    )
