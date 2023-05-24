from game import runGames, trainPlayer
from ghostAgents import GhostAgent, GhostAgentSlightlyRandom, GhostsAgent, GhostsAgentSample
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent, DQNAgent, ImitationAgent, ImitationAgent, QNet
from searchAgents import MaxScoreAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout
from util import Vector2d
import pickle

if __name__ == "__main__":
    # ghostsAgent = GhostsAgent(4)
    map_size = Vector2d(15, 15)
    expertAgent  = MaxScoreAgent()
    playerAgent = ImitationAgent(QNet(map_size), expertAgent)
    playerAgent = pickle.load(open("ImitationAgent.pkl", "rb"))
    ghosts_pos = []
    player_pos = Vector2d(7, 7)
    ghostsAgent = [GhostAgent(i) for i in range(1, 6)]
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghostNum = 5,
        player_pos = player_pos,
        ghosts_pos = ghosts_pos,
    )
    try:
        trainPlayer(
            display=NullGraphics,
            layout=layout,
            player=playerAgent,
            ghosts=ghostsAgent,
            numTrain=100000
        )
    except KeyboardInterrupt:
        print("Training stopped by user.")
    pickle.dump(playerAgent, open("ImitationAgent.pkl", "wb"))

    runGames(
        display=NullGraphics,
        layout=layout,
        player=playerAgent,
        ghosts=ghostsAgent,
        numGames=3
    )
