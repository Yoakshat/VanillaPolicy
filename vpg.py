import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import ale_py

# Code masters CartPole-v1

ENV_NAME = "CartPole-v1"
# initializing environment
env = gym.make(ENV_NAME)
obs, _ = env.reset()
# print(obs.shape)

def createNetwork(layers, activation): 
    network = []

    for i in range(0, len(layers)-1): 
        linear = nn.Linear(layers[i], layers[i+1])

        network.append(linear)
        if(i+1 < len(layers) - 1):
            network.append(activation)

    return network

# policy dim: dimension of action space
# value dim: 1 number
def createPolicyValue(layers, numActions, activation = nn.Tanh()):
    policy = createNetwork(layers + [numActions], activation)
    policy.append(nn.Softmax(dim=-1))
    value = createNetwork(layers + [1], activation)

    return nn.Sequential(*policy), nn.Sequential(*value)

# dynamic programming to compute discounted rewards
# without discount (set to 1)
def makeDiscounted(rewards, disc): 
    # go backwards
    discounts = torch.zeros(rewards.shape[0])
    discounts[-1] = rewards[-1]

    for i in range(len(rewards) - 2, -1, -1):
        discounts[i] = rewards[i] + disc * discounts[i+1]

    return discounts


class VGNBuffer: 
    # immediate rewards, logprobs, and values
    def __init__(self, numTraj, numSteps, discount): 
        dim = (numTraj, numSteps)
        self.dim = dim
        self.discount = discount

        # states dimension: (numTraj, numSteps, )
        self.states = torch.zeros((numTraj, numSteps, obs.shape[0]))
        self.actions = torch.zeros(dim, dtype=torch.long)
        self.rews = torch.zeros(dim)
        self.values = torch.zeros(dim)
        self.advantages = torch.zeros(dim)
        self.discRews = torch.zeros(dim)
        

    def add(self, t, s, state, act, rew, val): 
        self.states[t][s] = state
        self.actions[t][s] = act
        self.rews[t][s] = rew
        self.values[t][s] = val

    # pad for tens1 to match same shape as tens2
    def pad1D(self, tens, size): 
        zeroPadding = torch.zeros(size - tens.shape[0])
        return torch.cat((tens, zeroPadding))

    # computes advantage and discounted rewards
    # for trajectory
    def compute(self, t, gamma=0.99):
        # advantage = Q(s, a) - V(s)
        # = r(s, a) + V(s') - V(s)
        advTraj = self.rews[t][:-1] + gamma * (self.values[t][1:]) - self.values[t][:-1]

        # add padding if episode ends early
        self.advantages[t] = self.pad1D(advTraj, self.dim[1])
        self.discRews[t] = makeDiscounted(self.rews[t], self.discount) 


# policy + value network creation
hidden_layers = [64, 32]
policy, value = createPolicyValue([obs.shape[0]] + hidden_layers
                                  , env.action_space.n)

polOpt = Adam(policy.parameters(), lr=1e-3)
valOpt = Adam(value.parameters())


# value network is used to compute advantages in policy loss (reward_now + V' - V) 
# policy loss = (log prob) * (reward_now + V' - V)
# anything V should be thought of as a constant

# value loss = (values - disc_rewards)**2
# disc_rewards should be treated as a constant 
    

def updatePolicy(buff, valueNet): 
    numTraj = buff.dim[0]

    # softmax on last dimension (actions)
    actProbs = policy(buff.states)
    m = Categorical(probs=actProbs)
    # actions must be an int tensor
    logprobs = m.log_prob(buff.actions)

    # adv will be 0 (for timesteps we shouldn't consider)
    # loss = -1 * torch.sum((logprobs * buff.advantages)) / numTraj
    rew = buff.advantages
    if(not valueNet): 
        rew = buff.discRews

    # how this loss works
    # if higher reward, really try to push that probability towards 1
    loss = -1 * torch.sum((logprobs * rew)) / numTraj
    
    
    polOpt.zero_grad()
    loss.backward()
    
    # Update parameters
    polOpt.step()

    return loss.item()

# how many steps were actually in episode (e.g may have terminated early)
def updateValue(buff, totalSteps): 
    # will give [50, 100, 1] so squeeze end dimension
    vals = torch.squeeze(value(buff.states), -1) 

    loss = torch.sum((vals - buff.discRews)**2)/totalSteps
    valOpt.zero_grad()
    loss.backward()
    valOpt.step()

    return loss.item()

epochs = 100
trajectorySteps = 500
numTrajectories = 50
valueIters = 5

# total return of the trajectory
def train(valueNet = True, discount = 0.8):
    env = gym.make(ENV_NAME)
    obs, _ = env.reset()

    returns = []

    # actions per epoch

    for e in range(epochs):
        visActions = [] 
        totalTimeSteps = 0
        # define a new buffer every epoch
        buffer = VGNBuffer(numTrajectories, trajectorySteps, discount)
        avgReturn = 0

        for t in range(numTrajectories):
            timesteps = 0
            terminated = False

            while timesteps < trajectorySteps and not terminated:
                # collect trajectory
                with torch.no_grad():
                    stateTorch = torch.from_numpy(obs)

                    # here dimension is just (number of actions) so softmax(dim = 0)
                    action_probs = policy(stateTorch)
    

                    m = Categorical(action_probs)
                    action = m.sample().item()
                    visActions.append(action)


                    obs, reward, terminated, truncated, _ = env.step(action)
                    avgReturn += reward

                    
                    val = 0
                    if(valueNet): 
                        val = value(stateTorch).item()

                    # add in buffer at every timestep
                    buffer.add(t, timesteps, stateTorch, action, reward, val)

                timesteps += 1

            totalTimeSteps += timesteps

            # if terminated -> reset observation
            if(terminated): 
                obs, _ = env.reset()
            
            # compute advantages and discounted rewards for trajectory
            buffer.compute(t)

        avgReturn /= numTrajectories
        returns.append(avgReturn)


        # print("training")
        # now update

        # look at gradients for policyNet
        updatePolicy(buffer, valueNet)
 
        if(valueNet): 
            for _ in range(valueIters): 
                updateValue(buffer, totalTimeSteps)

    # plot returns
    plt.plot(returns)
    plt.ylabel("Average Return Over Trajectory")
    plt.xlabel("Number of Epochs")
    plt.savefig("Return.png")

    plt.close()

    torch.save(policy.state_dict(), "policyNet")


# train(valueNet = False, discount=0.99)

# actually, can you test model in real time? (and have it play)
def test_model(PATH, episodes=5): 
    env = gym.make(ENV_NAME, render_mode = "human")
    obs, _ = env.reset()
    pol, _ = createPolicyValue([obs.shape[0]] + hidden_layers
                                  , env.action_space.n)
    pol.load_state_dict(torch.load(PATH, weights_only=True))
    pol.eval()

    with torch.no_grad(): 
        for _ in range(episodes): 
            terminated = False
            steps = 0
            while not terminated: 
                stateTorch = torch.from_numpy(obs)

                action_probs = pol(stateTorch)
                m = Categorical(action_probs)
                action = m.sample().item()
                
                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1

                # never truncates
                if(truncated): 
                    print("Truncated")
            
            obs, _ = env.reset()

# why is avg return so high in training, but not when we test
test_model("policyNet")
