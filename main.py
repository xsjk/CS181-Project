from game import runGames, trainPlayer
from ghostAgents import GhostAgent, GhostAgentSlightlyRandom, GhostsAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent, DQNAgent
from searchAgents import MaxScoreAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout
from util import Vector2d
import pickle

if __name__ == "__main__":
    # ghostsAgent = GhostsAgent(4)
    map_size = Vector2d(15, 15)
    # playerAgent = DQNAgent(map_size)
    # playerAgent = QLearningAgent()
    ghosts_pos = []
    player_pos = Vector2d(1, 6)
    playerAgent = GreedyAgent()
    # playerAgent = pickle.load(open("QLearningAgent.pkl", "rb"))
    # ghostsAgent = [GhostAgent(i) for i in range(1, 6)]
    ghostsAgent = [GhostAgent(i) for i in range(1,6)]
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghostNum = 5,
        player_pos = player_pos,
        ghosts_pos = ghosts_pos,
    )

    # trainPlayer(
    #     display=NullGraphics,
    #     layout=layout,
    #     player=playerAgent,
    #     ghosts=ghostsAgent,
    #     numTrain=1000
    # )
    # pickle.dump(playerAgent, open("QLearningAgent.pkl", "wb"))
    # pickle.dump(playerAgent, open("DQNAgent.pkl", "wb"))
    # pickle.dump(playerAgent, open("SarsaLambdaAgent.pkl", "wb"))

    runGames(
        display=PygameGraphics,
        layout=layout,
        player=playerAgent,
        ghosts=ghostsAgent,
        numGames=3
    )
