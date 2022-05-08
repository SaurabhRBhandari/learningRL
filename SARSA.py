import random
import gym
import numpy as np
env=gym.make('FrozenLake-v0')

class SARSA():
    def __init__(self,env,epsilon,alpha,gamma,episodes,timesteps):
        self.env=env
        self.epsilon=epsilon
        self.alpha=alpha
        self.gamma=gamma
        self.n_episodes=episodes
        self.n_timesteps=timesteps
        self.n_actions=env.action_space.n
        self.n_states=env.observation_space.n
        self.Q=np.zeros((self.n_states,self.n_actions))
        
    def epsilon_greedy_policy(self,state):
        
        if random.uniform(0,1)<self.epsilon:
            return self.env.action_space.sample()
        
        else:
            return np.argmax(self.Q[state])
    
    def train(self):
        for i in range(self.n_episodes):
            s=self.env.reset()
            a=self.epsilon_greedy_policy(s)
            for t in range(self.n_timesteps):
                s_,r,done,_=self.env.step(a)
                a_=self.epsilon_greedy_policy(s_)
                self.Q[s][a] += self.alpha*(r+self.gamma*self.Q[s_][a_]-self.Q[s][a])
                s=s_
                a=a_
                if done:
                    break
model=SARSA(env,epsilon=0.8,alpha=0.85,gamma=0.9,episodes=50000,timesteps=1000)
model.train()
print(model.Q)        


