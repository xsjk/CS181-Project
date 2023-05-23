import pickle
from reinforcementAgents import DQNAgent

playerAgent:DQNAgent = pickle.load(open("DQNAgent.pkl", "rb"))
print(playerAgent.alpha)
print(playerAgent.epsilon)
print(playerAgent.gamma)
print(playerAgent.lambd)
print(playerAgent.batch_size)
print(playerAgent.memory_size)
print(len(playerAgent.memory))

