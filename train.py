from game import runGames, Game
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
from collections import deque
from environment import Environment, NaiveRewardEnvironment, BFSRewardEnvironment
from game import Agent
from game import ClassicGameRules
from rich.progress import track

import pickle
from torch.utils.tensorboard import SummaryWriter


import pkgutil
if pkgutil.find_loader("rich"):
    from rich import traceback
    traceback.install()
if pkgutil.find_loader("torch"):
    import torch
    from deepLearningAgents import OneHotDQNAgent, FullyConnectedDQNAgent ,ImitationAgent, AutoPriorityReplayBuffer



def trainPlayer(
    map_size: Vector2d,
    ghost_num: int,
    layoutGenerator: LayoutGenerator,
    player: Agent,
    ghosts: list[Agent],
    envType: type = NaiveRewardEnvironment,
    numTrain: int = 100,
    scoreChange: list[int] = [1,125,750,-500],
):
    rules = ClassicGameRules()
    layout = layoutGenerator.generate(map_size=map_size, ghost_num=ghost_num)
    display = NullGraphics(map_size, None)

    for _ in track(range(numTrain), description="Training..."):
        layout.arrangeAgents(layout.player_pos, layout.ghosts_pos)
        # print(layout.agentPositions)
        game: Game = rules.newGame(
            layout, player, ghosts, display, scoreChange, 
        )
        env: Environment = envType(player, startState=game.state)
        player.train(env)

        scores = env.state.getScore()
        wins = env.state.isWin()
        print(f"Score: {scores}, Win: {wins}, Epsilon: {player.epsilon}")
    player.epsilon = 0.0
    return player


if __name__ == "__main__":
    # ghostsAgent = SmartGhostsAgent(4)
    map_size = Vector2d(15, 15)
    ghost_num = 5
    expertAgent = MaxScoreAgent()
    playerAgent = OneHotDQNAgent(map_size)
    # playerAgent = pickle.load(open("OneHotDQNAgent.pkl", "rb"))
    # playerAgent.writer = SummaryWriter("runs/OneHotDQNAgent")
    playerAgent.epsilon_min = 0.1
    playerAgent.epsilon_decay = 1e-5
    # playerAgent.memory = AutoPriorityReplayBuffer(playerAgent.memory_size, playerAgent.abs_td_error)
    # playerAgent.optimizer = torch.optim.RMSprop(playerAgent.model.parameters(), lr=0.0025, alpha=0.95, eps=0.01)
    # playerAgent.target_model = deepcopy(playerAgent.model)
    # playerAgent.target_model.eval()
    # playerAgent.target_model.requires_grad_(False)
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
        playerAgent.writer.flush()
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
    pickle.dump(playerAgent, open("OneHotDQNAgent.pkl", "wb"))