from dataclasses import dataclass
from agentRules import Agent
from game import runGames
from train import trainPlayer
from ghostAgents import GreedyGhostAgent, GhostAgentSlightlyRandom, SmartGhostsAgent, GhostsAgentSample
from playerAgents import RandomAgent
from multiAgents import TimidAgent, AlphaBetaAgent, ExpectimaxAgent
from reinforcementAgents import MCTSAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent
from searchAgents import MaxScoreAgent
from display import NullGraphics
from layout import Layout
from util import Vector2d
from copy import deepcopy
from collections import deque
import pickle
from torch.utils.tensorboard import SummaryWriter
from rich.progress import track
from rich import traceback
traceback.install()
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from time import sleep, time
from pprint import pprint
from environment import NaiveRewardEnvironment
from abc import abstractmethod, ABC
from typing import Callable


np.set_printoptions(threshold=np.inf)
COLOUR_LIST = ['g', 'r', 'b', 'brown']


@dataclass
class EA(ABC):

    DNA_idx: list[str]
    fitnessEvaluator: Callable[dict, float]
    pop_size: int = 50
    n_kid: int = 75
    times: int = 20
    pop: dict[str, np.ndarray] = None
    
    EvolutionHistroy = np.array([])


    def __post_init__(self):        

        filelist = os.listdir()
        for file in filelist:
            if file[:6] == "figure":
                os.remove(file)

        
    @abstractmethod
    def initialize(self):
        raise NotImplementedError
        
    def gen_init_DNA(self) -> np.ndarray:
        f = True
        while f:
            init_DNA = np.random.rand(1, self.DNA_size)
            print('Initial DNA: ', pd.DataFrame(
                init_DNA[0], index=self.DNA_idx, columns=['']), '\n')
            if input('是否重新生成?(y/n) ') == 'n':
                f = False
            else:
                f = True
        return init_DNA

    @property
    def DNA_size(self):
        return len(self.DNA_idx)

    def getUnfitnesses(self) -> np.ndarray[float]:
        '''DNA是一个[ , , ,]'''
        unfitnesses = []
        DNAs = self.pop['DNA']
        for i in range(len(DNAs)):
            mean = self.getUnfitness(DNAs[i])
            print('\r', i, DNAs[i], mean)
            unfitnesses.append(mean)
        return np.array(unfitnesses)
        # return np.array([run_once(*DNA) for DNA in DNAs])

    @property
    def EvolutionHistroyData(self):
        return pd.DataFrame(self.EvolutionHistroy, columns=self.DNA_idx)


    def evolute(self):
        raise NotImplementedError


    def update(self):
        assert self.pop is not None, "Make sure you have initialzed the population"

        print('#'*20)
        print('GENERATION:', len(self.EvolutionHistroy))
        print('#'*20, '\n')

        self.evolute()

        self.EvolutionHistroy = np.vstack(
            (self.EvolutionHistroy, np.mean(self.pop['DNA'], 0)))

        self.plot()

        self.save()

        print()

    def plot(self):

        plt.cla()
        plt.axis(self.axis)
        plt.title('进化史（当前为第{}代）'.format(len(self.EvolutionHistroy)),
                  fontproperties='SimHei', size=15)
        plt.xlabel('Generation 代', fontproperties='SimHei', size=12)
        plt.ylabel('param 参数', fontproperties='SimHei', size=12)
        for i, c in zip(range(self.EvolutionHistroy.shape[1]), COLOUR_LIST):
            plt.plot(list(range(len(self.EvolutionHistroy))),
                     self.EvolutionHistroy[:, i], c=c, label=self.DNA_idx[i])
        plt.legend()
        plt.savefig('figure{}.png'.format(len(self.EvolutionHistroy)))
        plt.pause(1)

    def mainloop(self, N_GENERATIONS=None):
        self.N = N_GENERATIONS
        if N_GENERATIONS == None:
            self.axis = [0, len(self.EvolutionHistroy), -0.1, 1.1]
            # self.plot()
            print('无限进化\n')
            while True:
                self.axis = [0, len(self.EvolutionHistroy), -0.1, 1.1]
                self.update()
        else:
            self.axis = [0, N_GENERATIONS +
                        len(self.EvolutionHistroy)-1, -0.1, 1.1]
            # self.plot()
            print('即将进化{}代\n'.format(N_GENERATIONS))
            for _ in track(range(N_GENERATIONS)):
                self.update()

    @property
    def unfitnessData(self):
        return pd.DataFrame(
            np.hstack((self.pop['DNA'],
                       np.expand_dims(self.pop['Fitness'], axis=1))),
            columns=self.DNA_idx+['Time'])

    @abstractmethod
    def getUnfitness(self, DNA):
        raise NotImplementedError
    

class GA(EA):
    """
    Genetic Algorithm
    """

    def initialize(self, init_DNA=None):
        
        if init_DNA is None:
            init_DNA=self.gen_init_DNA()
        else:
            print('您输入的初始DNA为:')
            print('Initial DNA: ', pd.DataFrame(
                init_DNA, index=self.DNA_idx, columns=['']), '\n')

        self.EvolutionHistroy = np.array([init_DNA])

        self.pop = {'DNA': np.repeat(init_DNA, self.pop_size, axis=0),
                    'MTR': np.random.rand(self.pop_size, self.DNA_size),
                    'Fitness': np.full(self.pop_size, np.nan)}
        pass


    def evolute(self):

        kids = {
            'DNA': np.zeros((self.n_kid, self.DNA_size)),
            'MTR': np.zeros((self.n_kid, self.DNA_size))
        }

        self.crossover(kids)
        self.mutate(kids)
        self.add(kids)
        self.selection()


    def crossover(self, kids):
        for DNA, MTR in zip(kids['DNA'], kids['MTR']):
            father, mother = np.random.choice(np.arange(self.pop_size), size=2, replace=False)
            cp = np.random.randint(0, 2, self.DNA_size, dtype=bool)
            DNA[cp] = self.pop['DNA'][father, cp]
            DNA[~cp] = self.pop['DNA'][mother, ~cp]
            MTR[cp] = self.pop['MTR'][father, cp]
            MTR[~cp] = self.pop['MTR'][mother, ~cp]

    def mutate(self, kids):
        DNA = kids['DNA']
        MTR = kids['MTR']
        MTR[:] = np.maximum(MTR + (np.random.rand(*MTR.shape)-0.5), 0.)
        DNA[:] = 1/(1+np.exp(-np.random.randn(*DNA.shape)
                    * MTR-np.log(DNA/(1-DNA))))
        DNA[:] = np.clip(DNA[:], 1e-6, 1-1e-6)
        # DNA[:] = 1/(1+np.exp((-np.random.randn(*DNA.shape)+DNA)*MTR))

    def add(self, kids):
        for key in ['DNA', 'MTR']:
            self.pop[key] = np.vstack((self.pop[key], kids[key]))
    
    def selection(self):

        self.pop['Fitness'] = self.getUnfitnesses()
        idxs = self.pop['Fitness'].argsort()[:self.pop_size]

        for key in self.pop.keys():
            self.pop[key] = self.pop[key][idxs]

    def getUnfitness(self, DNA) -> float:
        params = {k: v for k, v in zip(self.DNA_idx ,DNA)}
        print(params)
        return self.fitnessEvaluator(params)

class DE(EA):
    """
    Differential Evolution
    """
    def evolute(self):

        pass

class CCEA(EA):
    """
    Cooperative Co-Evolution Algorithm
    """
    def evolute(self):
        # subpopulations = initialize_subpopulations()
        # evaluate_subpopulations()
        # 对每个子种群进行个体评估
        evaluate_subpopulations(subpopulations)

        # 在子种群之间共享信息或交换个体
        exchange_information(subpopulations)

        # 在子种群内进行进化操作
        evolve_subpopulations(subpopulations)

        pass



class EDA(EA):
    """
    Estimation of Distribution Algorithm
    """
    pass


if __name__ == '__main__':
    ea = GA(["Alpha", "Gamma", "Epsillon"])
    ea.initialize()
    ea.mainloop()
    # print()
    # print(ea.unfitnessData.sort_values(by='Time'))
    # print(ea.unfitnessData.mean())


    exit(0)

    # ghostsAgent = SmartGhostsAgent(4)
    map_size = Vector2d(15, 15)
    expertAgent  = MaxScoreAgent()
    playerAgent = OneHotDQNAgent(map_size)
    playerAgent = pickle.load(open("OneHotDQNAgent.pkl", "rb"))
    # playerAgent.writer = SummaryWriter("runs/OneHotDQNAgent")
    playerAgent.epsilon_min = 0.1
    playerAgent.epsilon_decay = 1e-5
    # playerAgent.memory = AutoPriorityReplayBuffer(playerAgent.memory_size, playerAgent.abs_td_error)
    # playerAgent.optimizer = torch.optim.RMSprop(playerAgent.model.parameters(), lr=0.0025, alpha=0.95, eps=0.01)
    # playerAgent.target_model = deepcopy(playerAgent.model)
    # playerAgent.target_model.eval()
    # playerAgent.target_model.requires_grad_(False)
    # playerAgent.memory = deque(maxlen=playerAgent.memory_size)
    # playerAgent.batch_size = 10

    ghosts_pos = []
    player_pos = None
    ghostsAgent = [GreedyGhostAgent(i) for i in range(1, ghost_num+1)]
    layout = Layout(
        map_size = map_size,
        tile_size = (30,30),
        ghost_num = 4,
        player_pos = player_pos,
        ghosts_pos = ghosts_pos,
    )
    try:
        playerAgent.writer.flush()
        trainPlayer(
            layout=layout,
            player=playerAgent,
            ghosts=ghostsAgent,
            numTrain=1000000
        )
    except KeyboardInterrupt:
        print("Training stopped by user.")
    finally:
        playerAgent.writer.close()
    pickle.dump(playerAgent, open("OneHotDQNAgent.pkl", "wb"))

