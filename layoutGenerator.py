import pickle
import numpy as np
import itertools
from abc import ABC, abstractmethod
from searchAgents import MaxScoreAgent
from display import NullGraphics
from util import auto_convert, type_check, Vector2d, Singleton
from layout import Layout
from game import ClassicGameRules
from ghostAgents import GreedyGhostAgent
import os
from rich.progress import track


class LayoutGenerator(ABC):

    @abstractmethod
    def generate(self, map_size: Vector2d, ghost_num: int) -> Layout:
        raise NotImplementedError

LayoutGenerator.__init__ = auto_convert(verbose=False)(LayoutGenerator.__init__)


class RandomLayoutGenerator(LayoutGenerator, metaclass=Singleton):

    def generate(self, map_size: Vector2d, ghost_num: int) -> Layout:
        player_pos = Vector2d(*np.random.randint(1, map_size.x+1, 2))
        available_pos = set(itertools.product(range(1, map_size.x+1), range(1, map_size.y+1))) - {player_pos}
        assert player_pos not in available_pos
        available_pos = list(available_pos)
        np.random.shuffle(available_pos)
        ghosts_pos = available_pos[:ghost_num]
        assert player_pos not in ghosts_pos
        return Layout(map_size=map_size, ghost_num=ghost_num, player_pos=player_pos, ghosts_pos=ghosts_pos)


class SpecialLayoutGenerator(RandomLayoutGenerator, metaclass=Singleton):

    samples: dict[tuple[Vector2d, int], Layout] = {}

    def __init__(self):
        if os.path.exists("SpecialLayoutGenerator.pkl"):
            self.samples.update(pickle.load(open("SpecialLayoutGenerator.pkl","rb")))

    def generate(self, map_size: Vector2d, ghost_num: int) -> Layout:
        if (map_size, ghost_num) in self.samples:
            return np.random.choice(self.samples[(map_size, ghost_num)])
        else:
            self.prepare(map_size, ghost_num, 1)
            return self.generate(map_size, ghost_num)

    def pregenerate(self, map_size: Vector2d, ghost_num: int) -> Layout:
        expertAgent = MaxScoreAgent()
        ghostsAgent = [GreedyGhostAgent(i) for i in range(1, ghost_num+1)]
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
    
    @auto_convert
    def prepare(self, map_size: Vector2d, ghost_num: int, num: int):
        for _ in track(range(num)):
            sample = self.pregenerate(map_size, ghost_num)
            self.samples.setdefault((map_size, ghost_num), [])
            self.samples[(map_size, ghost_num)].append(sample)
            pickle.dump(self.samples, open("SpecialLayoutGenerator.pkl", "wb"))

if __name__ == "__main__":
    
    generator = SpecialLayoutGenerator()
    generator.prepare((15,15), 4, 100)
    # generator.prepare(Vector2d(15,15), 5, 10)
    layout = generator.generate(Vector2d(15,15), 4)
    print(layout)

