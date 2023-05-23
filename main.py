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
    map_size = Vector2d(10, 10)
    # playerAgent = DQNAgent(map_size)
    # playerAgent = QLearningAgent()
    ghosts_pos = [(1,1),(1,4),(2,2),(4,1),(4,4)]
    player_pos = Vector2d(3, 6)
    playerAgent = AlphaBetaAgent()
    # playerAgent = pickle.load(open("SarsaLambdaAgent.pkl", "rb"))
    ghostsAgent = [GhostAgent(i) for i in range(1, 4)]
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghostNum = 3,
        player_pos = player_pos,
        ghosts_pos = ghosts_pos,
    )

    # trainPlayer(
    #     display=PygameGraphics,
    #     layout=layout,
    #     player=playerAgent,
    #     ghosts=ghostsAgent,
    #     numTrain=1000
    # )
    # pickle.dump(playerAgent, open("QLearningAgent.pkl", "wb"))
    # # pickle.dump(playerAgent, open("DQNAgent.pkl", "wb"))
    # pickle.dump(playerAgent, open("SarsaLambdaAgent.pkl", "wb"))

    runGames(
        display=PygameGraphics,
        layout=layout,
        player=playerAgent,
        ghosts=ghostsAgent
    )
