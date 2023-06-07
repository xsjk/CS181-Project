import numpy as np
import itertools
from abc import ABC, abstractmethod
from searchAgents import MaxScoreAgent
from display import NullGraphics
from util import auto_convert, type_check, Vector2d
from layout import Layout
from game import ClassicGameRules
from ghostAgents import GreedyGhostAgent


class LayoutGenerator(ABC):

    @abstractmethod
    def generate(self):
        raise NotImplementedError

LayoutGenerator.__init__ = auto_convert(verbose=False)(LayoutGenerator.__init__)


class RandomLayoutGenerator(LayoutGenerator):

    def generate(self, map_size: Vector2d, ghost_num: int) -> Layout:
        player_pos = Vector2d(*np.random.randint(1, map_size.x+1, 2))
        available_pos = set(itertools.product(range(1, map_size.x+1), range(1, map_size.y+1))) - {player_pos}
        available_pos = list(available_pos)
        np.random.shuffle(available_pos)
        ghosts_pos = available_pos[:ghost_num]
        return Layout(map_size=map_size, ghost_num=ghost_num, player_pos=player_pos, ghosts_pos=ghosts_pos)


class SpecialLayoutGenerator(RandomLayoutGenerator):

    def generate(self, map_size: Vector2d, ghost_num: int) -> Layout:
        expertAgent = MaxScoreAgent()
        ghostsAgent = [GreedyGhostAgent(i) for i in range(1, 6)]
        scoreChange: list[int] = [1,125,750,-500]
        gameDisplay = NullGraphics(map_size, None)

        while True:
            layout = super().generate(map_size=map_size, ghost_num=ghost_num)
            rules = ClassicGameRules()
            layout.arrangeAgents(layout.player_pos, layout.ghosts_pos)
            game = rules.newGame(layout, expertAgent, ghostsAgent, gameDisplay, scoreChange, True)
            game.run()
            if game.state.isWin():
                return layout


if __name__ == "__main__":
    
    generator = SpecialLayoutGenerator()
    layout = generator.generate(Vector2d(15,15), 5)
    print(layout)

