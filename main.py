from game import runGames
from ghostAgents import GhostAgent, GhostAgentSlightlyRandom
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent
from searchAgents import LongestLiveAgent
from displayModes import PygameGraphics
from layout import Layout

if __name__ == "__main__":
    # ghostsAgent = GhostsAgent(4)
    runGames(
        display=PygameGraphics,
        layout=Layout(
            map_size = (20,20),
            tile_size = (30,30),
            ghostNum = 7,
            player_pos = (20,20),
            ghosts_pos = [(1,1),(2,1),(3,1),(4,1)]
        ),
        player=KeyboardAgent(),
        ghosts=[GhostAgentSlightlyRandom(i+1) for i in range(7)]
    )
