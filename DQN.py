from collections import deque
import numpy as np
import copy
from torch import nn
import random
import torch
import gym
from tqdm import tqdm

env=gym.make('MsPacman-v0')

def state_to_tensor(state):
    np.reshape(state, (1, 210,160,3)).transpose(0,3,1,2)/255
    state = np.reshape(state, (1, 210,160,3)).transpose(0,3,1,2)/255
    state_tensor = torch.from_numpy(state).float()
    return state_tensor

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.stack=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool2d(3,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=1792, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=9),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.stack(x)

    
class DQN():
    def __init__(self,buffer_size,gamma,epsilon,update_rate, device="cpu"):

        self.replay_buffer=deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.device = device
        self.gamma=gamma
        self.epsilon=epsilon
        self.update_rate=update_rate
        self.main_network=self.build_network().to(self.device)
        self.target_network=self.build_network().to(self.device)
        self.target_network=copy.deepcopy(self.main_network)
        self.loss_fn=nn.MSELoss()
        self.optimizer=torch.optim.Adam(self.main_network.parameters(),lr=1e-3)
        self.env=env
        
    def build_network(self):
        return CNN()
    
    def store_transition(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))

    def epsilon_greedy_policy(self,state,epsilon):
        if random.uniform(0,1)<epsilon:
            return self.env.action_space.sample()
        else:
            self.main_network.to(self.device)
            B = state_to_tensor(state).to(self.device)
            A = self.main_network(B)
            Q_vals = A.detach().to("cpu").numpy()
            return np.argmax(Q_vals)
    
    def fit(self,batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        self.main_network.train()
        self.loss_fn = self.loss_fn.to(self.device)
        for state, action, reward, next_state, done in minibatch:
            stat = state_to_tensor(state)
            stat = stat.to(self.device)
            next_stat = state_to_tensor(next_state)
            next_stat = next_stat.to(self.device)
            self.target_network.to(device=self.device)
            self.main_network.to(device=self.device)
            if not done:
                A=self.target_network(next_stat)
                Q_ = A.detach().to(device="cpu").numpy()
                target_Q = reward + self.gamma*Q_
            else:
                continue
    
            Q_values=self.main_network(stat).to("cpu")
            
            loss=self.loss_fn(Q_values,torch.from_numpy(target_Q))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def update_target_network(self):
        self.target_network=copy.deepcopy(self.main_network)
    
    def train(self):
        num_episodes=500
        num_timesteps=20000
        batch_size=1024
        done=False
        for i in tqdm(range(num_episodes)):
            time_steps=0
            Return=0
            state=env.reset()
            for t in tqdm(range(num_timesteps)):
                env.render()
                time_steps +=1
                if time_steps % self.update_rate ==0:
                    self.update_target_network()
                action = self.epsilon_greedy_policy(state,self.epsilon)
                next_state, reward, done, _ =env.step(action)
                self.store_transition(state,action,reward,next_state,done)
                state=next_state
                Return += reward
                if done:
                    print('Episode: ',i,',','Return',Return)
                    break
                if(len(self.replay_buffer)>batch_size):
                    self.fit(batch_size)
                    self.replay_buffer = deque([], maxlen=self.buffer_size)
            state=env.reset()
                
model=DQN(buffer_size=5000,gamma=0.9,epsilon=0.8,update_rate=1000, device="cuda")
model.train()
        
        
    