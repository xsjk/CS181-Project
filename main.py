from game import runGames
from ghostAgents import GhostAgent
from playerAgents import KeyboardAgent, RandomAgent
from multiAgents import ReflexAgent, AlphaBetaAgent, ExpectimaxAgent
from displayModes import PygameGraphics, NullGraphics
from layout import Layout
from vector import Vector2d

if __name__ == "__main__":
    layout = Layout(
        width = 10,
        height = 10,
        tile_width = 30,
        tile_height = 30,
        ghostNum = 4,
        player_pos = Vector2d(5,5),
        ghosts_pos = [Vector2d(1,1),Vector2d(5,4)]
    ) 

    runGames(
        display=PygameGraphics,
        layout=layout,
        player=KeyboardAgent(), 
        ghosts=[GhostAgent(i+1) for i in range(4)],
    )
