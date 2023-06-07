from game import runGames, trainPlayer
from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent, ApproximateQAgent, MyFeatures
from display import NullGraphics
from layout import Layout
from util import Vector2d
import pickle


import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback
    traceback.install()
if pkgutil.find_loader("torch"):
    import torch
    from deepLearningAgents import OneHotDQNAgent, FullyConnectedDQNAgent ,ImitationAgent, AutoPriorityReplayBuffer

if __name__ == "__main__":
    map_size = Vector2d(15, 15)
    playerAgent = ApproximateQAgent(MyFeatures())
    # playerAgent = pickle.load(open("ApproximateQAgent.pkl", "rb"))

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
        pickle.dump(playerAgent, open("ApproximateQAgent.pkl", "wb"))
