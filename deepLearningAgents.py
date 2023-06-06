from typing import Optional
import torch
from torch import nn, optim
import torch.nn.functional as F
from reinforcementAgents import QLearningAgent, SarsaLambdaAgent
from util import Vector2d, type_check
from game import Action, GameState, Agent
from abc import ABC, abstractmethod
from environment import Environment
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from collections import deque
from dataclasses import dataclass
import heapq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

@dataclass
class ReplayBuffer:
    maxlen: int

    def __post_init__(self):
        self.memory = deque(maxlen=self.maxlen)
    def append(self, item):
        self.memory.append(item)
    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)


@dataclass
class PriorityReplayBuffer(ReplayBuffer):

    def __post_init__(self):
        self.memory = np.empty(self.maxlen, dtype=object)
        self.weights = np.zeros(self.maxlen)
        self.indices = [] # priority queue
        self.index = 0

    def append(self, weight, item):
        if self.index < self.maxlen:
            heapq.heappush(self.indices, (weight, self.index))
            self.memory[self.index] = item
            self.weights[self.index] = weight
            self.index += 1
        else:
            weight_, index_ = self.indices[0]
            if weight_ < weight:
                heapq.heapreplace(self.indices, (weight, index_))
                self.memory[index_] = item
                self.weights[index_] = weight

    def sample(self, batch_size: int):
        indices = [i for w, i in random.choices(self.indices, weights=self.weights[:self.index], k=batch_size)]
        return self.memory[indices]
    
    def take(self, batch_size):
        indices = heapq.nlargest(batch_size, self.indices)
        return self.memory[indices]
    
    def __len__(self):
        return self.index

@dataclass
class AutoPriorityReplayBuffer(PriorityReplayBuffer):
    evaluator: nn.Module = None
    def append(self, item):
        super().append(self.evaluator(*item), item)


@dataclass
class DQNAgent(QLearningAgent):

    model: nn.Module = None

    memory_size: int = 100000
    batch_size: int = 128
    gamma: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 1e-5
    epsilon_min: float = 0.01

    update_freq: int = 10
    update_counter: int = 0
    total_reward: float = 0.0

    illegal_action_penalty: float = -1.0
    tau: float = 0.01

    loss_fn: nn.Module = nn.MSELoss()

    def __post_init__(self):
        super().__init__()
        assert isinstance(self.model, nn.Module)

        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        self.target_model.requires_grad_(False)

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.01, alpha=self.alpha, eps=0.01)
        self.memory = ReplayBuffer(self.memory_size)
        # self.memory = AutoPriorityReplayBuffer(self.memory_size, self.abs_td_error)
        

    def getQValue(self, S, A: Action) -> float:
        return self.getQValues(S)[A.index]
    
    def getQValues(self, S: GameState):
        X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
        X = X.unsqueeze(0)
        ys = self.target_model(X)
        ys = ys.squeeze(0)
        return ys
        
    def getPolicy(self, S: GameState) -> Action:
        Qs = self.getQValues(S)
        legal = S.getLegalActionIndices()
        random.shuffle(legal)
        return Action.list[max(legal, key=lambda i: Qs[i], default=None)]

    def getAction(self, S: GameState) -> Action:
        X = torch.tensor(self.state2matrix(S), dtype=torch.float32).to(device)
        X = X.unsqueeze(0)
        ys = self.target_model(X)
        ys = ys.squeeze(0)
        legal = S.getLegalActionIndices()
        random.shuffle(legal)
        return Action.list[max(legal, key=lambda i: ys[i], default=None)]
    
    def td_error(self, S: np.ndarray, A: int, S_: np.ndarray, R: float, done: bool, legal: Optional[list[int]] = None) -> float:
        X = torch.tensor(S, dtype=torch.float32)
        X = X.unsqueeze(0)
        Q = self.target_model(X)
        Q = Q.squeeze(0)[A]

        X_ = torch.tensor(S_, dtype=torch.float32)
        X_ = X_.unsqueeze(0)
        Q_ = self.target_model(X_)
        Q_target = R
        if not done:
            if legal is not None:
                R += self.gamma * Q_.squeeze(0)[legal].max()
        error = Q - Q_target
        error = error.detach().cpu().numpy()
        
        assert not np.isnan(error)
        return error
    
    def abs_td_error(self, S, A, S_, R: float, done: bool, legal: Optional[list[int]] = None) -> float:
        
        error = self.td_error(S, A, S_, R, done, legal)
        # print(error)
        # if R > 0:
        #     error *= 1000
        return abs(error)

    def update(self, S, A, S_, R: float, done: bool):
        A = A.index
        legal = S.getLegalActionIndices()
        S = self.state2matrix(S)
        S_ = self.state2matrix(S_)
        self.memory.append((S, A, S_, R, done, legal))
        self.total_reward += R

        if len(self.memory) < self.batch_size:
            return
        
        self.update_counter += 1

        batch = self.memory.sample(self.batch_size)
        S_batch = torch.tensor([s for s, _, _, _, _, _ in batch], dtype=torch.float32)
        A_batch = torch.tensor([a for _, a, _, _, _, _ in batch], dtype=torch.long)
        S_batch_ = torch.tensor([s_ for _, _, s_, _, _, _ in batch], dtype=torch.float32)
        R_batch = torch.tensor([r for _, _, _, r, _, _ in batch], dtype=torch.float32)
        done_batch = torch.tensor([d for _, _, _, _, d, _ in batch], dtype=torch.bool)
        legal_batch = [l for _, _, _, _, _, l in batch]


        Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
        Q_batch_ = self.target_model(S_batch_)

        target = R_batch
        for i in range(self.batch_size):
            if not done_batch[i]:
                target[i] += self.gamma * Q_batch_[i,legal_batch[i]].max()
        
        loss = self.loss_fn(Q_batch, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_counter % self.update_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)
            self.writer.add_scalar('loss', loss, self.update_counter // self.update_freq)
            self.writer.add_scalar('reward', self.total_reward / self.update_counter, self.update_counter // self.update_freq)
            
            # print(f'loss: {loss}')
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            

    @staticmethod
    @abstractmethod
    def state2matrix(s: GameState) -> np.ndarray:
        raise NotImplementedError

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)



# class DDQN(nn.Module):
#     def __init__(self, n_actions, input_dim, fc1_dims, fc2_dims):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, fc1_dims)
#         self.fc2 = nn.Linear(fc1_dims, fc2_dims)
#         self.V = nn.Linear(fc2_dims, 1)
#         self.A = nn.Linear(fc2_dims, n_actions)
        
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()

#     def forward(self, state):
#         x = nn.Flatten()(state)
#         x = self.relu1(self.fc1(state))
#         x = self.relu2(self.fc2(x))

#         V = self.V(x)
#         A = self.A(x)

#         Q = V + (A - torch.mean(A, dim = 1, keepdim = True))

#         return Q

#     def advantage(self, state):
#         x = self.relu1(self.fc1(state))
#         x = self.relu2(self.fc2(x))

#         return self.A(x)


# @dataclass
# class D3QNAgent(QLearningAgent):

#     batch_size: int = 1024
#     epsilon_decay: float = 1e-8
#     epsilon_min: float = 0.01
#     mem_size: int = 1000000
#     fc1_dims: int = 128
#     fc2_dims: int = 128
#     replace: int = 100
#     loss_fn: nn.Module = nn.MSELoss()


#     def __ipost_init__(self):
#         self.update_counter = 0
#         self.memory = ReplayBuffer(maxlen=self.mem_size)
#         self.model = DDQN(n_actions=9, input_dim=15*15*3, fc1_dims=1024, fc2_dims=1024)
#         self.target_model = DDQN(n_actions=9, input_dim=15*15*3, fc1_dims=1024, fc2_dims=1024)

#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        

#     def update(self, S: GameState, A: Action, S_: GameState, R: float, done: bool):
#         self.memory.append((S, A, S_, R, done))

#         if len(self.memory) < self.batch_size:
#             return
        
#         if self.update_counter % self.replace == 0:
#             self.target_model.load_state_dict(self.model.state_dict())
        
#         batch = self.memory.sample(self.batch_size)
#         legal = S.getLegalActionIndices()
        
#         S_batch = torch.tensor([self.state2matrix(s) for s, _, _, _, _ in batch], dtype=torch.float32)
#         A_batch = torch.tensor([a.index for _, a, _, _, _ in batch], dtype=torch.long)
#         S_batch_ = torch.tensor([self.state2matrix(s_) for _, _, s_, _, _ in batch], dtype=torch.float32)
#         R_batch = torch.tensor([r for _, _, _, r, _ in batch], dtype=torch.float32)
#         done = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.bool)

#         Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
#         Q_batch_ = self.target_model(S_batch_) #.max(1)[0].detach()
#         # select max Q_ from legal actions 
#         random.shuffle(legal)
#         Q_batch_ = torch.tensor([
#             Q_batch_[i, legal].max(0)[1].item()
#             for i in range(len(batch))
#         ], dtype=torch.float32)
#         Q_target = R_batch + self.gamma * Q_batch_ * ~done

#         self.optimizer.zero_grad()
#         loss = self.loss_fn(Q_target, Q_batch)
#         loss.backward()
#         self.optimizer.step()

#         self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.eps_min else self.eps_min
#         self.update_counter += 1


class OneHotDQNAgent(DQNAgent):

    writer = SummaryWriter('runs/OneHotDQNAgent')

    @type_check
    def __init__(self, map_size: Vector2d):
        # super().__init__(model=nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(32 * map_size.x * map_size.y, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 9),
        # ))
        super().__init__(model=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
            
            nn.Linear(64 * map_size.x * map_size.y, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
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

class OneHotDQNAgent2(OneHotDQNAgent):

    writer = SummaryWriter('runs/OneHotDQNAgent2')

    @type_check
    def __init__(self, map_size: Vector2d):
        DQNAgent.__init__(self, model=nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Flatten(),
            
            nn.Linear(64 * map_size.x * map_size.y, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 9),
        ))



class MultiScaleNet(nn.Module):
    def __init__(self, map_size):
        super(MultiScaleNet, self).__init__()
        
        self.global_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        
        self.local_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        self.local_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(96 * map_size.x * map_size.y, 1024),  # 32+64 = 96
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 9),
        )
    
    def forward(self, x):
        global_feat = self.global_conv(x)
        local_feat1 = self.local_conv1(x)
        local_feat2 = self.local_conv2(local_feat1)
        
        combined_feat = torch.cat([global_feat, local_feat2], dim=1)
        flattened_feat = self.flatten(combined_feat)
        output = self.fc(flattened_feat)
        
        return output

class OneHotDQNAgent3(OneHotDQNAgent):

    writer = SummaryWriter('runs/OneHotDQNAgent3')

    @type_check
    def __init__(self, map_size: Vector2d):
        DQNAgent.__init__(self, model=MultiScaleNet(map_size))



class FullyConnectedDQNAgent(DQNAgent):

    writer = SummaryWriter('runs/FullyConnectedDQNAgent')

    @type_check
    def __init__(self, map_size: Vector2d):
        super().__init__(model=nn.Sequential(
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

class FullyConnectedDQNAgent2(DQNAgent):

    writer = SummaryWriter('runs/FullyConnectedDQNAgent2')
    
    @type_check
    def __init__(self, map_size: Vector2d):
        super().__init__(model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(map_size.x * map_size.y, 1024),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
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


    



class PosDQNAgent(DQNAgent):

    writer = SummaryWriter('runs/PosDQNAgent')

    @staticmethod
    def state2matrix(s: GameState) -> np.ndarray:
        arr = np.zeros((s.getNumAgents(), 2))
        player_pos = s.getPlayerPosition()
        arr[0][0] = player_pos.x
        arr[0][1] = player_pos.y
        for i, ghost in enumerate(s.getGhostStates()):
            pos = ghost.getPosition()
            arr[i + 1][0] = pos.x
            arr[i + 1][1] = pos.y
        return arr

class AttentionNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return torch.sum(x * self.attention(x), dim=1)

class AttentionPosRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(2, 256, batch_first=True)
        self.attention = AttentionNet(256)
        self.fc = nn.Linear(256, 9)
        self.hidden = None

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # output: (batch_size, input_size)
        # self.hidden: (1, batch_size, input_size)
        if self.training:
            output, hidden = self.rnn(x, self.hidden)
            self.hidden = hidden.detach()
            output = self.attention(output)
            output = self.fc(output)
            return output
        else:
            output, _ = self.rnn(x)
            output = self.attention(output)
            output = self.fc(output)
            return output


class AttentionPosDQNAgent(PosDQNAgent):
    
    writer = SummaryWriter('runs/AttentionPosDQNAgent')

    @type_check
    def __init__(self):
        super().__init__(model=AttentionPosRNN())
        
    def update(self, S, A, S_, R: float, done: bool):
        super().update(S, A, S_, R, done)
        if done:
            self.model.hidden = None

    def getPolicy(self, S: GameState) -> Action:
        hidden = self.model.hidden
        self.model.hidden = None
        A = super().getPolicy(S)
        self.model.hidden = hidden
        return A
    

class FixnumPosDQNAgent(PosDQNAgent):

    writer = SummaryWriter('runs/FixnumPosDQNAgent')

    @type_check
    def __init__(self, num_ghosts: int):
        super().__init__(model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * (num_ghosts + 1), 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        ))



class GCNnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim).to(device)
        self.conv2 = GCNConv(hidden_dim, hidden_dim).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = torch.mean(x, dim=1)  # Aggregate node features
        q_values = self.fc(x)
        return q_values
    
class GCNDQNAgent(PosDQNAgent):

    writer = SummaryWriter('runs/GCNDQNAgent')

    def __init__(self, num_ghosts: int):
        super().__init__(model=GCNnet(2, 128, 9))
        self.edge_index = torch.tensor(
            [[i, j] for i in range(num_ghosts + 1) for j in range(num_ghosts + 1) if i != j], 
            dtype=torch.long
        ).t()

    def update(self, S, A, S_, R: float, done: bool):
        legal = S.getLegalActionIndices()
        S = self.state2matrix(S)
        S_ = self.state2matrix(S_)
        A = A.index

        self.memory.append((S, A, S_, R, done, legal))
        if len(self.memory) < self.batch_size:
            return
        
        self.update_counter += 1

        batch = self.memory.sample(self.batch_size)
        S_batch = Data(
            x=torch.tensor([s for s, _, _, _, _, _ in batch], dtype=torch.float32),
            edge_index=self.edge_index,
        )
        A_batch = torch.tensor([a for _, a, _, _, _, _ in batch], dtype=torch.long)
        S_batch_ = Data(
            x=torch.tensor([s_ for _, _, s_, _, _, _ in batch], dtype=torch.float32),
            edge_index=self.edge_index,
        )
        R_batch = torch.tensor([r for _, _, _, r, _, _ in batch], dtype=torch.float32)
        done_batch = torch.tensor([d for _, _, _, _, d, _ in batch], dtype=torch.bool)
        legal_batch = [l for _, _, _, _, _, l in batch]

        Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
        Q_batch_ = self.target_model(S_batch_)
        
        target = R_batch
        for i in range(self.batch_size):
            if not done_batch[i]:
                target[i] += self.gamma * Q_batch_[i,legal_batch[i]].max()
        
        loss = self.loss_fn(Q_batch, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_counter % self.update_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.mul_(1.0 - self.tau)
                    target_param.data.add_(self.tau * param.data)
            self.writer.add_scalar('loss', loss, self.update_counter // self.update_freq)
            # print(f'loss: {loss}')

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
            
    def getQValues(self, S: GameState):
        X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
        X = X.unsqueeze(0)
        ys = self.target_model(Data(
            x=X,
            edge_index=self.edge_index,
        ))
        ys = ys.squeeze(0)
        return ys
     
    def td_error(self, S: np.ndarray, A: int, S_: np.ndarray, R: float, done: bool, legal: Optional[list[int]]) -> float:
        X = torch.tensor(S, dtype=torch.float32)
        X = X.unsqueeze(0)
        Q = self.target_model(Data(
            x=X,
            edge_index=self.edge_index,
        ))
        Q = Q.squeeze(0)[A]

        X_ = torch.tensor(S_, dtype=torch.float32)
        X_ = X_.unsqueeze(0)
        Q_ = self.target_model(Data(
            x=X_,
            edge_index=self.edge_index,
        ))
        Q_target = R
        if not done:
            if legal is not None:
                R += self.gamma * Q_.squeeze(0)[legal].max()
        error = Q - Q_target
        error = error.detach().cpu().numpy()
        return error
        
        
        

class ImitationAgent(OneHotDQNAgent):
    expert: Agent
    expert_loss_fn = nn.MSELoss()

    def __init__(self, map_size: Vector2d, expert: Agent):
        super().__init__(map_size)
        self.expert = expert

    def train(self, env: Environment):
        env.resetState()
        S = env.getCurrentState()
        A = self.getTrainAction(S)
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

            S_, R, done = env.takeAction(A)
            if done:
                break
            self.update(S, A, S_, R, done)
            S = S_
            A = self.getTrainAction(S)

    def getTrainAction(self, S: GameState) -> Action:
        with torch.no_grad():
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
            X = X.unsqueeze(0)
            ys = self.model(X)
            ys = ys.squeeze(0)
            legal = S.getLegalActionIndices()
            random.shuffle(legal)
            return Action.list[max(legal, key=lambda i: ys[i], default=None)]
        

class ActorCriticsAgent(FullyConnectedDQNAgent):
    actor: nn.Module
    critic: nn.Module
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    critic_loss_fn = nn.MSELoss()

    writer = SummaryWriter('runs/ActorCriticsAgent')


    def __init__(self, map_size: Vector2d):
        super().__init__(map_size)
        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(map_size.x * map_size.y, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 9),
            nn.Softmax(dim=1),
        )
        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(map_size.x * map_size.y, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        # target networks
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)
        self.actor_target.eval()
        self.critic_target.eval()
        self.actor_target.requires_grad_(False)
        self.critic_target.requires_grad_(False)

        self.update_counter = 0


    def update(self, S: GameState, A: Action, S_: GameState, R: float, done: bool):
        A: int = A.index
        S = self.state2matrix(S)
        S_ = self.state2matrix(S_)

        self.memory.append((S, A, S_, R, done))
        if len(self.memory) < self.batch_size:
            return
        
        self.update_counter += 1
        batch = self.memory.sample(self.batch_size)
        S_batch = torch.tensor([s for s, _, _, _, _ in batch], dtype=torch.float32)
        A_batch = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.int64)
        S_batch_ = torch.tensor([s_ for _, _, s_, _, _ in batch], dtype=torch.float32)
        R_batch = torch.tensor([r for _, _, _, r, _ in batch], dtype=torch.float32)
        done_batch = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.bool)

        # Update critic
        Q_batch = self.critic(S_batch).squeeze(1)
        Q_batch_ = self.critic_target(S_batch_).squeeze(1)
        Q_target = R_batch + self.gamma * Q_batch_ * ~done_batch
        td_delta = Q_target - Q_batch
        log_probs = torch.log(self.actor(S_batch).gather(1, A_batch.unsqueeze(1)) + 1e-8).squeeze(1)
        actor_loss = torch.mean(-log_probs * td_delta.detach()) # detach to stop gradient flow to critic network
        critic_loss = self.critic_loss_fn(Q_batch, Q_target)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step() 
        self.critic_optimizer.step()

        # Update target networks
        if self.update_counter % self.update_freq == 0:
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
                
            self.writer.add_scalar('actor_loss', torch.abs(actor_loss), self.update_counter // self.update_freq)
            self.writer.add_scalar('critic_loss', critic_loss, self.update_counter // self.update_freq)

            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def soft_update(self, target_network, source_network):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def getTrainAction(self, S: GameState) -> Action:
        with torch.no_grad():
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
            X = X.unsqueeze(0)
            ys = self.actor_target(X)
            ys = ys.squeeze(0)
            ys = F.softmax(ys, dim=0)
            
            # Filter out illegal actions and re-normalize the probabilities
            legal = S.getLegalActionIndices()
            ys_legal = torch.tensor([ys[i] for i in legal], dtype=torch.float32)
            ys_legal /= ys_legal.sum()
            
            dist = torch.distributions.Categorical(ys_legal)
            while True:
                i = dist.sample().item()
                if i in legal:
                    return Action.list[i]
    
    def getAction(self, S: GameState) -> Action:
        with torch.no_grad():
            X = torch.tensor(self.state2matrix(S), dtype=torch.float32, device=device)
            X = X.unsqueeze(0)
            self.actor.eval()
            ys = self.actor(X)
            self.actor.train()
            ys = ys.squeeze(0)
            ys = F.softmax(ys, dim=0)
            
            # Filter out illegal actions and re-normalize the probabilities
            legal = S.getLegalActionIndices()
            ys_legal = torch.tensor([ys[i] for i in legal], dtype=torch.float32)
            ys_legal /= ys_legal.sum()
            
            dist = torch.distributions.Categorical(ys_legal)
            while True:
                i = dist.sample().item()
                if i in legal:
                    return Action.list[i]


@dataclass
class DSLAgent(SarsaLambdaAgent):

    model: nn.Module = None

    epsilon_decay: float = 1e-5
    epsilon_min: float = 0.01
    tau: float = 0.01

    def __post_init__(self):
        assert isinstance(self.model, nn.Module)
        

        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        self.target_model.requires_grad_(False)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001)

        params = list(self.model.parameters())
        num_params = sum(p.numel() for p in params)
        print("Total number of parameters: ", num_params)

        self.eligibility_traces = {name: torch.zeros_like(param, requires_grad=False) for name, param in self.model.named_parameters()}
        
        self.update_counter = 0
        self.update_freq = 10


    def getPolicy(self, S: GameState) -> Action:
        Qs = self.getQValues(S)
        legal = S.getLegalActionIndices()
        random.shuffle(legal)
        return Action.list[max(legal, key=lambda i: Qs[i])]

    def getQValues(self, S: GameState) -> torch.Tensor:
        X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
        X = X.unsqueeze(0)
        ys = self.target_model(X)
        ys = ys.squeeze(0)
        return ys
    
    def getQValue(self, S: GameState, A: Action) -> float:
        return self.getQValues(S)[Action.index(A)].item()
    

    def update(self, S: GameState, A: Action, R: float, S_: GameState, A_: Action, done: bool):


        if done:
            for name, param in self.model.named_parameters():
                self.eligibility_traces[name].zero_()
            return
        

        self.update_counter += 1
        
        S = torch.tensor(self.state2matrix(S), dtype=torch.float32)
        A = A.index
        S_ = torch.tensor(self.state2matrix(S_), dtype=torch.float32)
        A_ = A_.index

        Q = self.model(S.unsqueeze(0)).squeeze(0)[A]
        Q_ = self.target_model(S_.unsqueeze(0)).squeeze(0)[A_]

        Q_target = R + self.gamma * Q_ * ~done
        td_error = Q_target - Q

        # Get gradients
        self.model.zero_grad()
        Q.backward()

        # Update eligibility traces
        for name, param in self.model.named_parameters():
            self.eligibility_traces[name] = self.gamma * self.lambd * self.eligibility_traces[name] + param.grad

        # Update weights
        for name, param in self.model.named_parameters():
            param.data.add_(self.alpha * td_error * self.eligibility_traces[name])


        if self.update_counter % self.update_freq == 0:
            self.soft_update(self.target_model, self.model)
            loss = torch.abs(td_error).item()
            self.writer.add_scalar('loss', loss, self.update_counter)
            # print(f'loss: {loss}')
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def soft_update(self, target_network, source_network):
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    @staticmethod
    @abstractmethod
    def state2matrix(s: GameState) -> np.ndarray:
        raise NotImplementedError
    

class FullyConnectedDSLAgent(DSLAgent):

    writer = SummaryWriter('runs/FullyConnectedDSLAgent')

    @type_check
    def __init__(self, map_size: Vector2d):
        super().__init__(model=nn.Sequential(
            nn.Flatten(),
            nn.Linear(map_size.x * map_size.y, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
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




if __name__ == "__main__":
    pass
