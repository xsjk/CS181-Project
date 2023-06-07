from run import runGames
from train import trainPlayer
from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from playerAgents import RandomAgent
from multiAgents import TimidAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent
from searchAgents import MaxScoreAgent
from display import NullGraphics
from environment import NaiveRewardEnvironment, NullRewardEnvironment
from layout import Layout
from layoutGenerator import LayoutGenerator, RandomLayoutGenerator, SpecialLayoutGenerator
from util import Vector2d
from copy import deepcopy
from environment import Environment, NaiveRewardEnvironment, BFSRewardEnvironment
from collections import deque
import pickle
from torch.utils.tensorboard import SummaryWriter

import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback
    traceback.install()
if pkgutil.find_loader("torch"):
    import torch
    from deepLearningAgents import DQNAgent, OneHotDQNAgent, FullyConnectedDQNAgent ,AutoPriorityReplayBuffer


if __name__ == "__main__":
    # ghostsAgent = SmartGhostsAgent(4)
    map_size = Vector2d(15, 15)
    ghost_num = 4
    expertAgent  = MaxScoreAgent()
    playerAgent = FullyConnectedDQNAgent(map_size)
    # playerAgent = pickle.load(open("FullyConnectedDQNAgent.pkl", "rb"))
    # playerAgent.writer = SummaryWriter("runs/FullyConnectedDQNAgent")
    # playerAgent.optimizer = torch.optim.Adam(playerAgent.model.parameters(), lr=0.1)
    # playerAgent.memory = AutoPriorityReplayBuffer(playerAgent.memory_size, playerAgent.abs_td_error)
    playerAgent.epsilon_min = 0.1
    playerAgent.epsilon_decay = 1e-5
    # playerAgent.optimizer = torch.optim.RMSprop(playerAgent.model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
    # playerAgent.target_model = deepcopy(playerAgent.model)
    # playerAgent.target_model.eval()
    # playerAgent.target_model.requires_grad_(False)
    # playerAgent.memory = deque(maxlen=playerAgent.memory_size)
    # playerAgent.batch_size = 10
    ghosts_pos = []
    player_pos = None
    ghostsAgent = [GreedyGhostAgent(i) for i in range(1, ghost_num+1)]
    try:
        trainPlayer(
            envType=BFSRewardEnvironment,
            map_size=map_size,
            ghost_num=ghost_num,
            layoutGenerator=SpecialLayoutGenerator(),
            player=playerAgent,
            ghosts=ghostsAgent,
            numTrain=1000000
        )
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        playerAgent.writer.close()
    pickle.dump(playerAgent, open("FullyConnectedDQNAgent.pkl", "wb"))