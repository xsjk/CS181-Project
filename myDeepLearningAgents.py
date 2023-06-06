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
from collections import deque, namedtuple
from dataclasses import dataclass
import heapq



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


@dataclass
class DQNAgent(QLearningAgent):

    model: nn.Module

    memory_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    alpha: float = 0.001
    epsilon: float = 1.0
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01

    update_freq: int = 10
    update_counter: int = 0
    tau: float = 0.01

    loss_fn: nn.Module = nn.MSELoss()

    writer: SummaryWriter = None

    def __post_init__(self):
        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        self.target_model.requires_grad_(False)

        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.alpha, alpha=0.95, eps=0.01)
        self.memory = deque(maxlen=self.memory_size)
        self.priority = deque(maxlen=self.memory_size)

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
        return Action.list[max(legal, key=Qs.__getitem__)]
    

    def get_td_error(self, S: np.ndarray, A: int, S_: np.ndarray, R: float, done: bool):
        S = torch.tensor(S, dtype=torch.float32)
        S_ = torch.tensor(S_, dtype=torch.float32)
        A = torch.tensor(A, dtype=torch.long)
        R = torch.tensor(R, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        Qs = self.target_model(S)
        Qs_ = self.target_model(S_)
        Q = Qs[A]
        Q_ = torch.max(Qs_)
        Q_target = R if done else R + self.gamma * Q_
        td_error = Q_target - Q
        return td_error.item()
    
    def update(self, S: GameState, A: Action, S_: GameState, R: float, done: bool):
        
        A = A.index
        S = self.state2matrix(S)
        S_ = self.state2matrix(S_)
        self.memory.append((S, A, S_, R, done))
        self.priority.append(self.get_td_error(S, A, S_, R, done))
        if len(self.memory) < self.batch_size:
            return
        
        self.update_counter += 1

        batch_indices = np.random.choice(len(self.memory), self.batch_size, p=np.array(self.priority) / sum(self.priority))
        batch = [self.memory[i] for i in batch_indices]
        S_batch = torch.tensor([s for s, _, _, _, _ in batch], dtype=torch.float32)
        A_batch = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long)
        S_batch_ = torch.tensor([s_ for _, _, s_, _, _ in batch], dtype=torch.float32)
        R_batch = torch.tensor([r for _, _, _, r, _ in batch], dtype=torch.float32)
        done_batch = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.bool)

        Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
        Q_batch_ = self.target_model(S_batch_).max(1)[0].detach()
        target = R_batch + self.gamma * Q_batch_ * ~done_batch

        
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

    writer = SummaryWriter('runs/OneHotDQNAgent')

    @type_check
    def __init__(self, map_size: Vector2d):
        # super().__init__(nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(32 * map_size.x * map_size.y, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 9),
        # ))
        super().__init__(nn.Sequential(
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
        DQNAgent.__init__(self, nn.Sequential(
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
        DQNAgent.__init__(self, MultiScaleNet(map_size))



class FullyConnectedDQNAgent(DQNAgent):

    writer = SummaryWriter('runs/FullyConnectedDQNAgent')

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

class FullyConnectedDQNAgent2(DQNAgent):

    writer = SummaryWriter('runs/FullyConnectedDQNAgent2')
    
    @type_check
    def __init__(self, map_size: Vector2d):
        super().__init__(nn.Sequential(
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
        super().__init__(AttentionPosRNN())
        
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
        super().__init__(nn.Sequential(
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
        super().__init__(GCNnet(2, 128, 9))
        self.edge_index = torch.tensor(
            [[i, j] for i in range(num_ghosts + 1) for j in range(num_ghosts + 1) if i != j], 
            dtype=torch.long
        ).t()

    def update(self, S, A, S_, R: float, done: bool):
        S = self.state2matrix(S)
        S_ = self.state2matrix(S_)
        A = A.index

        self.memory.append((S, A, S_, R, done))
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[1:]
        elif len(self.memory) < self.batch_size:
            return
        
        self.update_counter += 1

        batch = random.sample(self.memory, self.batch_size)
        S_batch = Data(
            x=torch.tensor([s for s, _, _, _, _ in batch], dtype=torch.float32),
            edge_index=self.edge_index,
        )
        A_batch = torch.tensor([a for _, a, _, _, _ in batch], dtype=torch.long)
        S_batch_ = Data(
            x=torch.tensor([s_ for _, _, s_, _, _ in batch], dtype=torch.float32),
            edge_index=self.edge_index,
        )
        R_batch = torch.tensor([r for _, _, _, r, _ in batch], dtype=torch.float32)
        done_batch = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.bool)

        Q_batch = self.model(S_batch).gather(1, A_batch.unsqueeze(1)).squeeze(1)
        Q_batch_ = self.target_model(S_batch_).max(1)[0].detach()
        target = R_batch + self.gamma * Q_batch_ * ~done_batch
        
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

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
    def getQValues(self, S: GameState):
        X = torch.tensor(self.state2matrix(S), dtype=torch.float32)
        X = X.unsqueeze(0)
        ys = self.target_model(Data(
            x=X,
            edge_index=self.edge_index,
        ))
        ys = ys.squeeze(0)
        return ys
     
        
        
        

class ImitationAgent(OneHotDQNAgent):
    expert: Agent
    expert_loss_fn = nn.MSELoss()

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
        batch = random.sample(self.memory, self.batch_size)
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

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

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


class DSLAgent(SarsaLambdaAgent):

    writer: SummaryWriter

    def __init__(self, net, **kwargs):
        super().__init__(**kwargs)
        self.model = net
        self.tau = 0.01

        self.target_model = deepcopy(self.model)
        self.target_model.eval()
        self.target_model.requires_grad_(False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        params = list(self.model.parameters())
        num_params = sum(p.numel() for p in params)
        print("Total number of parameters: ", num_params)

        self.eligibility_traces = {name: torch.zeros_like(param, requires_grad=False) for name, param in self.model.named_parameters()}
        
        self.update_counter = 0
        self.update_freq = 10

        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

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
            self.writer.add_scalar('loss', torch.abs(td_error).item(), self.update_counter)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

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
        super().__init__(nn.Sequential(
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
