import numpy as np
import torch
from torch import nn
import random
import gym
import time

class NeuralNetwork(nn.Module):
    def __init__(self,in_features,out_features):
        super(NeuralNetwork,self).__init__()
        
        self.stack=nn.Sequential(
            nn.Linear(in_features,64),
            nn.ReLU(),
            nn.Linear(64,out_features),
            nn.Softmax(dim=1),
        )
        
    def forward(self,x):
        return self.stack(x)

GAMMA = 0.9            # discount factor
LR = 1e-2              # learning rate 

class Agent():
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.state_size = state_size
        
        self.policy_net = NeuralNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        
        self.saved_log_probs = []
        self.rewards = []
        
    
    def step(self, reward, log_probs):
        
        self.rewards.append(reward)
        self.saved_log_probs.append(log_probs)
        
                
    def act(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state).cpu()
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action)
        
    def reset(self):
        
        self.saved_log_probs = []
        self.rewards = []
        
    def learn(self, gamma):
        
        discounts = [gamma**i for i in range(len(self.rewards)+1)]
        
        R = sum([a*b for a,b in zip(discounts, self.rewards)])
        
        policy_loss = []
        
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

env=gym.make('CartPole-v0')
agent=Agent(state_size=env.observation_space.shape[0],action_size=env.action_space.n)

num_iters=10000
for i in range(num_iters):
    state = env.reset()
    agent.reset()
    step = 0
    R=0
    while True:
        action, log_probs = agent.act(state.copy())
        state, reward, done, _ = env.step(action)
        step += 1
        reward = -100 if done and step < 450 else reward
        R=R+reward  
        agent.step(reward, log_probs)
        if done:
            break
        if i%100==0:
            env.render()
    if i%1000==0:
        print(f"Episode:{i},Return{R}")
    agent.learn(GAMMA)

env.close()