from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import ReflexAgent, AlphaBetaAgent, ExpectimaxAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout
from vector import Vector2d as V

if __name__ == "__main__":
    layout = Layout(
        map_size = V(30,40),
        tile_size = V(20,10),
        ghostNum = 4,
        player_pos = V(5,5),
        ghosts_pos = [V(1,1),V(5,4)]
    ) 

    runGames(
        display=PygameGraphics,
        layout=layout,
        player=KeyboardAgent(), 
        ghosts=[GhostAgent(i+1) for i in range(4)],
    )
