import torch
from torch import nn, unsqueeze
import torch.nn.functional as F
import gym
env=gym.make('CartPole-v0')

class NeuralNetwork(nn.Module):
    def __init__(self,state_size,action_size):
        super(NeuralNetwork,self).__init__()
        self.action_size=action_size
        self.stack=nn.Sequential(
            nn.Linear(state_size,32),
            nn.ReLU(),
            nn.Linear(32,action_size),
        )
        
    def forward(self,x):
        x=torch.from_numpy(x).float().unsqueeze(0)
        if self.action_size == 1:
            return self.stack(x)
        else:
            return F.softmax(self.stack(x),dim=1)
GAMMA=0.99
class Agent():
    def __init__(self, state_size, action_size, alpha=0.0001, beta=0.0005):
        self.action_size = action_size
        self.state_size = state_size
        
        self.actor_net = NeuralNetwork(state_size, action_size)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=alpha)
        
        self.critic_net = NeuralNetwork(state_size, 1)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=beta)
        
        self.log_probs = None
        
                
    def act(self, state):
        
        probs = self.actor_net(state)
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        self.log_probs = action_probs.log_prob(action)
        
        return action.item()
        
    def step(self, state, reward, new_state, done):        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        cv_new_state = self.critic_net(new_state)
        cv_state = self.critic_net(state)
        
        reward = torch.tensor(reward, dtype=torch.float)
        
        delta = reward + GAMMA * cv_new_state * (1-int(done)) - cv_state
        
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2
        
        (actor_loss + critic_loss).backward()
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()

agent=Agent(env.observation_space.shape[0],env.action_space.n)

num_iters=10000
for i in range(num_iters):
    state = env.reset()
    score = 0
    while True:
        action = agent.act(state.copy())
        next_state, reward, done, _ = env.step(action)
        score+=reward
        reward = -100 if done and score < 450 else reward
        agent.step(state,reward,next_state,done)
        if done:
            break
        if i%100==0:
           env.render()
    if i%1000==0:
        print(f"Episode:{i},Score{score}")

env.close()
        
        