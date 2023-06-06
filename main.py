from game import runGames, trainPlayer
from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from playerAgents import RandomAgent
from multiAgents import TimidAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent
from searchAgents import MaxScoreAgent
from display import NullGraphics
from layout import Layout
from util import Vector2d
from copy import deepcopy
import pickle


import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback, print
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
    playerAgent = PygameKeyboardAgent()
    # playerAgent = MCTSAgent()
    # playerAgent = RandomAgent()
    # playerAgent = MaxScoreAgent()
    playerAgent = pickle.load(open("OneHotDQNAgent.pkl", "rb"))
    playerAgent = pickle.load(open("OneHotDQNAgent2.pkl", "rb"))
    playerAgent = pickle.load(open("OneHotDQNAgent3.pkl", "rb"))
    playerAgent = pickle.load(open("FullyConnectedDQNAgent.pkl", "rb"))
    playerAgent = pickle.load(open("FullyConnectedDQNAgent2.pkl", "rb"))
    playerAgent = pickle.load(open("ImitationAgent.pkl", "rb"))
    playerAgent = pickle.load(open("ActorCriticsAgent.pkl", "rb"))
    playerAgent = pickle.load(open("AttentionPosDQNAgent.pkl", "rb"))
    playerAgent = pickle.load(open("FixnumPosDQNAgent.pkl", "rb"))
    playerAgent = pickle.load(open("GCNDQNAgent.pkl", "rb"))
    playerAgent.target_model = deepcopy(playerAgent.model)
    playerAgent.target_model.eval()
    playerAgent.target_model.requires_grad_(False)
    playerAgent.memory.clear()
    playerAgent.tau = 0.1
    pickle.dump(playerAgent, open("GCNDQNAgent.pkl", "wb"))
    exit()
    
    playerAgent = pickle.load(open("AttentionPosDQNAgent.pkl", "rb"))
    playerAgent = pickle.load(open("ImitationAgent.pkl", "rb"))
    # ghostsAgent = SmartGhostsAgent(4)
    ghostsAgent = [GreedyGhostAgent(i) for i in range(1, ghost_num+1)]
    # ghostsAgent = list(map(GreedyGhostAgent, range(1, ghost_num+1)))
    scoreChange = [50,250,1000,-1000] # 0-3: normal, kill, win, lose
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
        scoreChange=scoreChange,
        numGames=100
    )
