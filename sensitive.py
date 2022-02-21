import random
from isort import file
import torch
from env import TradeEnv
from model import PolicyNetwork

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
hidden_dim = 256
env = TradeEnv("GOLD.csv", "bitcoin.csv", "bitcoin_pre.txt","gold_pre.txt")
state_dim  = len(env.states)
action_dim = len(env.actions)
state = torch.load('checkpoints.pth.tar', 'cpu')
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net.load_state_dict(state['policy_net'], strict=False)

def eval():
    state = env.reset()
    while True:
        action = policy_net.get_action(state)
        state, reward, done = env.step(state, action)
        if done:
            return state[-1]

g_cost = 0
b_cost = 0
result = []
x1 = []
x2 = []
size = 10
for i in range(10):
    g_cost += 0.1
    b_cost = 0
    print(i, '...')
    for j in range(10):
        b_cost += 0.1
        env.bitcoin_cost = b_cost
        env.gold_cost = g_cost
        x1.append(g_cost)
        x2.append(b_cost)
        result.append(eval())

for i in range(100):
    print(i, '...')
    x = random.random()
    

    
with open('sensetive.txt', "w") as f:
    print(result, file=f)
    print(x1, file=f)
    print(x2, file=f)
