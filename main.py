from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import GreedyAgent, AlphaBetaAgent, ExpectimaxAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout

if __name__ == "__main__":
    layout = Layout(
        width = 20,
        height = 20,
        tile_width = 30,
        tile_height = 30,
        ghostNum = 4,
        player_pos = (5,5),
        ghosts_pos = [(1,1),(5,4)]
    )

    runGames(
        display=PygameGraphics,
        layout=layout,
        player=KeyboardAgent(), 
        ghosts=[GhostAgent(i+1) for i in range(4)],
    )
