import torch
from torch import nn
from reinforcementAgents import QLearningAgent
from util import Vector2d, type_check
from game import Action, GameState, Agent
from abc import ABC, abstractmethod
from environment import Environment
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class QNet(nn.Module):
    def __init__(self, map_size: Vector2d):
        super().__init__()
        self.map_size = map_size
        # input dim: [3, map_size.x, map_size.y]
        # output dim: [9]
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * map_size.x * map_size.y, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        )

    def forward(self, x):
        y = self.model(x)
        return y


class DQNAgent(QLearningAgent):
    def __init__(self, net: nn.Module):
        super().__init__()

        self.actionList = list(Action)

        # action encoding: [9]
        # N, S, E, W, NW, NE, SW, SE, TP, STOP

        # state encoding: [map_size.x, map_size.y]
        self.model = net

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.gamma = 0.99
        self.batch_size = 128
        self.memory = []
        self.memory_size = 10000
        self.update_freq = 1000
        self.update_counter = 0

    def getQValue(self, S, A: Action) -> float:
        index = self.actionList.index(A)
        return self.getQValues(S)[index]
    
    def getQValues(self, S: GameState):
        with torch.no_grad():
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
            X = X.unsqueeze(0)
            ys = self.model(X)
            ys = ys.squeeze(0)
            return ys
        
    def getPolicy(self, S: GameState) -> Action:
        Qs = self.getQValues(S)
        legal = S.getLegalActions()
        random.shuffle(legal)
        return max(legal, key=lambda a: Qs[self.actionList.index(a)], default=None)

    def getAction(self, S: GameState) -> Action:
        with torch.no_grad():
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32).to(device)
            X = X.unsqueeze(0)
            ys = self.model(X)
            ys = ys.squeeze(0)
            legal = S.getLegalActions()
            random.shuffle(legal)
            return max(legal, key=lambda a: ys[self.actionList.index(a)], default=None)

    def update(self, S, A, S_, R: float) -> None:
        self.memory.append((S, A, S_, R))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        S_batch = torch.tensor([self.state2matrix(s) for s, _, _, _ in batch], dtype=torch.float32)
        A_batch = torch.tensor([self.actionList.index(a) for _, a, _, _ in batch], dtype=torch.long)
        S_batch_ = torch.tensor([self.state2matrix(s_) for _, _, s_, _ in batch], dtype=torch.float32)
        R_batch = torch.tensor([r for _, _, _, r in batch], dtype=torch.float32)

        Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
        Q_batch_ = self.model(S_batch_).max(1)[0].detach()
        target = R_batch + self.gamma * Q_batch_

        loss = self.loss_fn(Q_batch, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @staticmethod
    @abstractmethod
    def state2matrix(s: GameState) -> np.ndarray:
        raise NotImplementedError

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)


class OneHotDQNAgent(DQNAgent):
    @type_check
    def __init__(self, map_size: Vector2d):
        super().__init__(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * map_size.x * map_size.y, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        ))

    @staticmethod
    def state2matrix(s: GameState) -> np.ndarray:
        # [3, map_size.x, map_size.y]
        map_size = s.getMapSize()
        mat = np.zeros((3, map_size.x, map_size.y))
        player_pos = s.getPlayerPosition()
        mat[0][player_pos.y - 1][player_pos.x - 1] = 1
        for ghost in s.getGhostStates():
            pos = ghost.getPosition()
            if ghost.dead:
                mat[2][pos.y - 1][pos.x - 1] = 1
            else:
                mat[1][pos.y - 1][pos.x - 1] = 1
        return mat


class FullyConnectedDQNAgent(DQNAgent):
    @type_check
    def __init__(self, map_size: Vector2d):
        super().__init__(nn.Sequential(
            nn.Flatten(),
            nn.Linear(map_size.x * map_size.y, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
        ))

    @staticmethod
    def state2matrix(s: GameState) -> np.ndarray:
        # 0: empty
        # 1: player
        # 2: ghost
        # 3: dead ghost
        map_size = s.getMapSize()
        mat = np.zeros((map_size.x, map_size.y))
        player_pos = s.getPlayerPosition()
        mat[player_pos.y - 1][player_pos.x - 1] = 1
        for ghost in s.getGhostStates():
            pos = ghost.getPosition()
            if ghost.dead:
                mat[pos.y - 1][pos.x - 1] = 3
            else:
                mat[pos.y - 1][pos.x - 1] = 2
        return mat


class ImitationAgent(OneHotDQNAgent):
    expert: Agent
    expert_loss_fn = nn.MSELoss()

    @type_check
    def __init__(self, map_size: Vector2d, expert: Agent):
        super().__init__(map_size)
        self.expert = expert

    def train(self, env: Environment):
        env.resetState()
        S = env.getCurrentState()
        while True:
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
            X = X.unsqueeze(0)
            Q = self.model(X)
            Q = Q.squeeze(0)
            A_expert = self.expert.getAction(S)
            Q_expert = torch.tensor(A_expert.onehot[:9], dtype=torch.float32)
            loss = self.expert_loss_fn(Q, Q_expert)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            A = self.getTrainAction(S)
            S_, R, done = env.takeAction(A)
            if done:
                break
            self.update(S, A, S_, R)
            S = S_
            A = self.getTrainAction(S)

    def getTrainAction(self, S: GameState) -> Action:
        with torch.no_grad():
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
            X = X.unsqueeze(0)
            ys = self.model(X)
            ys = ys.squeeze(0)
            legal = S.getLegalActions()
            random.shuffle(legal)
            return max(legal, key=lambda a: ys[self.actionList.index(a)], default=None)
        



if __name__ == "__main__":
    pass
