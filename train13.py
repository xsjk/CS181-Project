from run import runGames
from train import trainPlayer
from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent, ApproximateQAgent, MyFeatures
from display import NullGraphics
from layout import Layout
from layoutGenerator import LayoutGenerator, RandomLayoutGenerator, SpecialLayoutGenerator
from util import Vector2d
from environment import BFSRewardEnvironment
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
    ghost_num = 4
    playerAgent = ApproximateQAgent(MyFeatures())
    # playerAgent = pickle.load(open("ApproximateQAgent.pkl", "rb"))
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
        pickle.dump(playerAgent, open("ApproximateQAgent.pkl", "wb"))
