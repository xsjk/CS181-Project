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
from collections import deque
import pickle
from torch.utils.tensorboard import SummaryWriter

import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback
    traceback.install()
if pkgutil.find_loader("torch"):
    import torch
    from deepLearningAgents import OneHotDQNAgent2, FullyConnectedDQNAgent ,ImitationAgent, AutoPriorityReplayBuffer, FullyConnectedDSLAgent

if __name__ == "__main__":
    # ghostsAgent = SmartGhostsAgent(4)
    map_size = Vector2d(15, 15)
    expertAgent  = MaxScoreAgent()
    playerAgent = OneHotDQNAgent2(map_size)
    playerAgent = pickle.load(open("OneHotDQNAgent2.pkl", "rb"))
    # playerAgent.writer = SummaryWriter('runs/OneHotDQNAgent2')
    playerAgent.optimizer = torch.optim.Adam(playerAgent.model.parameters(), lr=0.001)
    playerAgent.epsilon_min = 0.1
    playerAgent.epsilon_decay = 1e-5
    # playerAgent.memory = AutoPriorityReplayBuffer(playerAgent.memory_size, playerAgent.abs_td_error)
    # playerAgent.optimizer = torch.optim.RMSprop(playerAgent.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
    # playerAgent.target_model = deepcopy(playerAgent.model)
    # playerAgent.target_model.eval()
    # playerAgent.target_model.requires_grad_(False)
    # playerAgent.batch_size = 10
    ghosts_pos = []
    player_pos = None
    ghostsAgent = [GreedyGhostAgent(i) for i in range(1, 6)]
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghost_num = 5,
        player_pos = player_pos,
        ghosts_pos = ghosts_pos,
    )
    try:
        trainPlayer(
            displayType=NullGraphics,
            layout=layout,
            player=playerAgent,
            ghosts=ghostsAgent,
            numTrain=1000000
        )
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        playerAgent.writer.close()
    pickle.dump(playerAgent, open("OneHotDQNAgent2.pkl", "wb"))