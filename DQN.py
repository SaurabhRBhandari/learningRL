from collections import deque
import numpy as np
import copy
from torch import nn
import random
import torch
import gym

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
    def __init__(self,buffer_size,gamma,epsilon,update_rate):

        self.replay_buffer=deque(maxlen=buffer_size)
        self.gamma=gamma
        self.epsilon=epsilon
        self.update_rate=update_rate
        self.main_network=self.build_network()
        self.target_network=self.build_network()
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
            Q_vals=self.main_network(state_to_tensor(state)).detach().numpy()
            return np.argmax(Q_vals)
    
    def fit(self,batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        self.main_network.train()
        for state, action, reward, next_state, done in minibatch:
            if not done:
                Q_=self.target_network(state_to_tensor(next_state)).detach().numpy()
                target_Q = reward + self.gamma*Q_
            else:
                continue
    
            Q_values=self.main_network(state_to_tensor(state))
            
            loss=self.loss_fn(Q_values,torch.from_numpy(target_Q))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def update_target_network(self):
        self.target_network=copy.deepcopy(self.main_network)
    
    def train(self):
        num_episodes=500
        num_timesteps=20000
        batch_size=8
        done=False
        for i in range(num_episodes):
            time_steps=0
            Return=0
            state=env.reset()
            for t in range(num_timesteps):
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
            state=env.reset()
                
model=DQN(buffer_size=5000,gamma=0.9,epsilon=0.8,update_rate=1000)
model.train()
        
        
    