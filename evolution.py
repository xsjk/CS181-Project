import pandas as pd
import numpy as np
import os
from func_timeout import FunctionTimedOut, func_set_timeout
import matplotlib.pyplot as plt
from time import sleep, time
from pprint import pprint
from environment import PlayerGameEnvironment


class EA:
    np.set_printoptions(threshold=np.inf)
    colourlist = ['g', 'r', 'b', 'brown']

    def __init__(self, *args, **kargs):
        if args != () or kargs != {}:
            print('start a new evolution')
            self.initialize(*args, **kargs)
        else:
            print('continue from break point')
            self.reload()

    def reload(self, filename='EvolutionData.json'):

        with open('EvolutionData.dat', 'r') as f:
            data = eval(f.read().replace('\n', ''))
        self.__dict__.update(data)
        self.set_timeout(self.time_out)

    def set_timeout(self, time_out):
        @func_set_timeout(time_out)
        def run_once(*args):
            br = self.BR(ENV(self.Map, None, 2), *args)
            return br.TimeSpent*br.EpisodesSpent
        self.run_once = lambda *args: run_once(*args)

    def initialize(self,
                   brain,
                   INIT_DNA=None,
                   POP_SIZE=50,
                   N_KID=75,
                   TIMES=20,
                   time_out=10):

        filelist = os.listdir()
        for file in filelist:
            if file[:6] == "figure":
                os.remove(file)

        self.POP_SIZE = POP_SIZE
        self.N_KID = N_KID
        self.TIMES = TIMES
        self.brain = brain
        self.time_out = time_out
        with open("maze_map.dat", "r") as f:
            self.Map = eval(f.read())

        print('MAP:')
        print(pd.DataFrame(self.Map))
        print()

        if INIT_DNA == None:
            f = True
            while f:
                INIT_DNA = np.random.rand(1, self.DNA_SIZE)
                print('Initial DNA: ', pd.DataFrame(
                    INIT_DNA[0], index=self.DNA_IDX, columns=['']), '\n')
                if input('是否重新生成?(y/n) ') == 'n':
                    f = False
                else:
                    f = True
        else:
            INIT_DNA = np.array([INIT_DNA])
            print('您输入的初始DNA为:')
            print('Initial DNA: ', pd.DataFrame(
                INIT_DNA[0], index=self.DNA_IDX, columns=['']), '\n')

        self.EvolutionHistroy = INIT_DNA[:]

        self.pop = {'DNA': INIT_DNA.repeat(self.POP_SIZE, axis=0),
                    'MTR': np.random.rand(self.POP_SIZE, self.DNA_SIZE),
                    'Fitness': np.full(self.POP_SIZE, np.nan)}
        # self.mainloop()

        self.set_timeout(time_out)

    @property
    def BR(self):
        if self.brain == 'QLearning':
            return Brain.QLearning
        elif self.brain == 'Sarsa':
            return Brain.Sarsa
        elif self.brain == 'SarsaLambda':
            return Brain.SarsaLambda

    @property
    def DNA_IDX(self):
        if self.brain == 'SarsaLambda':
            return ['Alpha', 'Gamma', 'Epsilon', 'Lambda']
        else:
            return ['Alpha', 'Gamma', 'Epsilon']

    @property
    def DNA_SIZE(self):
        return len(self.DNA_IDX)

    def run(self, *args):
        unfitnesses = []
        for i in range(self.TIMES):
            r = self.run_once(*args)
            unfitnesses.append(r)
            print('\r', '{}/{}'.format(i+1, self.TIMES), 'Unfitness', r, end='')
        print('\r', end='')
        return np.mean(unfitnesses)

    def getUnfitness(self):
        '''DNA是一个[ , , ,]'''
        unfitnesses = []
        DNAs = self.pop['DNA']
        for i in range(len(DNAs)):
            try:
                mean = self.run(*DNAs[i])
            except FunctionTimedOut:
                mean = np.nan
            print('\r', i, DNAs[i], mean)
            unfitnesses.append(mean)
        return np.array(unfitnesses)
        # return np.array([run_once(*DNA) for DNA in DNAs])

    def mutate(self, kids):
        DNA = kids['DNA']
        MTR = kids['MTR']
        MTR[:] = np.maximum(MTR + (np.random.rand(*MTR.shape)-0.5), 0.)
        DNA[:] = 1/(1+np.exp(-np.random.randn(*DNA.shape)
                    * MTR-np.log(DNA/(1-DNA))))
        DNA[:] = np.clip(DNA[:], 1e-6, 1-1e-6)
        # DNA[:] = 1/(1+np.exp((-np.random.randn(*DNA.shape)+DNA)*MTR))

    @property
    def EvolutionHistroyData(self):
        return pd.DataFrame(self.EvolutionHistroy, columns=self.DNA_IDX)

    def evolute(self):
        kids = {'DNA': np.zeros((self.N_KID, self.DNA_SIZE)),
                'MTR': np.zeros((self.N_KID, self.DNA_SIZE))}

        for DNA, MTR in zip(kids['DNA'], kids['MTR']):
            father, mother = np.random.choice(
                np.arange(self.POP_SIZE), size=2, replace=False)
            cp = np.random.randint(0, 2, self.DNA_SIZE,
                                   dtype=np.bool)  # crossover points
            DNA[cp] = self.pop['DNA'][father, cp]
            DNA[~cp] = self.pop['DNA'][mother, ~cp]
            MTR[cp] = self.pop['MTR'][father, cp]
            MTR[~cp] = self.pop['MTR'][mother, ~cp]

        self.mutate(kids)

        for key in ['DNA', 'MTR']:
            self.pop[key] = np.vstack((self.pop[key], kids[key]))

        self.pop['Fitness'] = self.getUnfitness()
        idxs = self.pop['Fitness'].argsort()[:self.POP_SIZE]

        for key in self.pop.keys():
            self.pop[key] = self.pop[key][idxs]

    def save(self):
        data = self.__dict__.copy()
        del data['run_once']
        data = str(data)
        with open('EvolutionData.dat', 'w') as f:
            f.write(str(data))
        # pprint(self.__dict__)

    def main(self):
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
        for i, c in zip(range(self.EvolutionHistroy.shape[1]), self.colourlist):
            plt.plot(list(range(len(self.EvolutionHistroy))),
                     self.EvolutionHistroy[:, i], c=c, label=self.DNA_IDX[i])
        plt.legend()
        plt.savefig('figure{}.png'.format(len(self.EvolutionHistroy)))
        plt.pause(1)

    def mainloop(self, N_GENERATIONS=None):
        self.N = N_GENERATIONS

        if N_GENERATIONS == None:
            self.axis = [0, len(self.EvolutionHistroy), -0.1, 1.1]
            self.plot()
            print('无限进化\n')
            while True:
                self.axis = [0, len(self.EvolutionHistroy), -0.1, 1.1]
                self.main()
        else:
            self.axis = [0, N_GENERATIONS +
                         len(self.EvolutionHistroy)-1, -0.1, 1.1]
            self.plot()
            print('即将进化{}代\n'.format(N_GENERATIONS))
            for _ in range(N_GENERATIONS):
                self.main()

    @property
    def unfitnessData(self):
        return pd.DataFrame(
            np.hstack((self.pop['DNA'],
                       np.expand_dims(self.pop['Fitness'], axis=1))),
            columns=self.DNA_IDX+['Time'])


if __name__ == '__main__':
    ea = EA()
    ea.mainloop()
    print()
    print(ea.unfitnessData.sort_values(by='Time'))
    # print(ea.unfitnessData.mean())
