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
    from deepLearningAgents import ActorCriticsAgent, FullyConnectedDQNAgent ,AutoPriorityReplayBuffer, AttentionPosDQNAgent

if __name__ == "__main__":
    # ghostsAgent = SmartGhostsAgent(4)
    map_size = Vector2d(15, 15)
    expertAgent  = MaxScoreAgent()
    playerAgent = AttentionPosDQNAgent()
    playerAgent = pickle.load(open("AttentionPosDQNAgent.pkl", "rb"))
    # playerAgent.writer = SummaryWriter("runs/AttentionPosDQNAgent")
    # playerAgent.memory = AutoPriorityReplayBuffer(playerAgent.memory_size, playerAgent.abs_td_error)
    playerAgent.epsilon_min = 0.1
    playerAgent.epsilon_decay = 1e-5
    # playerAgent.optimizer = torch.optim.Adam(playerAgent.model.parameters(), lr=0.1)
    # playerAgent.memory = deque(maxlen=playerAgent.memory_size)
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
    pickle.dump(playerAgent, open("AttentionPosDQNAgent.pkl", "wb"))