from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent
from searchAgents import LongestLiveAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout

if __name__ == "__main__":
    runGames(
        display=PygameGraphics,
        layout=Layout(
            map_size = (6,6),
            tile_size = (30,30),
            ghostNum = 2,
            player_pos = (2,2),
            ghosts_pos = [(1,1),(1,4)]
        ),
        player=MCTSAgent(),
        ghosts=[GhostAgent(i+1) for i in range(2)],
    )
