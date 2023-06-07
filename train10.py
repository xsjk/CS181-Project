from game import runGames
from train import trainPlayer
from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from playerAgents import RandomAgent
from multiAgents import TimidAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent
from searchAgents import MaxScoreAgent
from display import NullGraphics
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
    from deepLearningAgents import ActorCriticsAgent, FullyConnectedDQNAgent ,AutoPriorityReplayBuffer, FullyConnectedDSLAgent

if __name__ == "__main__":
    # ghostsAgent = SmartGhostsAgent(4)
    map_size = Vector2d(15, 15)
    expertAgent  = MaxScoreAgent()
    ghost_num = 5
    playerAgent = FullyConnectedDSLAgent(map_size)
    # playerAgent.writer = SummaryWriter("runs/FullyConnectedDSLAgent")
    # playerAgent = pickle.load(open("FullyConnectedDSLAgent.pkl", "rb"))
    playerAgent.epsilon_decay = 1e-5
    playerAgent.epsilon_min = 0.1
    ghosts_pos = []
    player_pos = None
    ghostsAgent = [GreedyGhostAgent(i) for i in range(1, 6)]
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
    pickle.dump(playerAgent, open("FullyConnectedDSLAgent.pkl", "wb"))
