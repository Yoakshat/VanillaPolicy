import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim
import numpy as np


env = gym.make("ALE/SpaceInvaders-ram-v5")
obs, info = env.reset()

# actor network
hidden_layers = [64, 32]
all_layers = [obs.shape[0]] + hidden_layers + [env.action_space.n]

def createPolicy(layers, activation = nn.Tanh()): 
    network = []

    for i in range(0, len(layers)-1): 
        linear = nn.Linear(layers[i], layers[i+1])

        network.append(linear)
        if(i+1 < len(layers) - 1):
            network.append(activation)

    network.append(nn.Softmax(dim=1))
        
    return nn.Sequential(*network)

def create_actor(): 
    env = gym.make("ALE/SpaceInvaders-ram-v5")
    obs, info = env.reset()

    return env, np.expand_dims(obs, axis=0)

def fillActors(num_actors, envs, states): 
    while len(envs) < num_actors: 
        env, state = create_actor()
        envs.append(env)
        states.append(state)

    return envs, states

# num_actors is equivalent to batch size
def train(policy, optimizer, num_actors, num_epochs, update_timesteps):
    envs, states = fillActors(num_actors, [], [])

    for e in range(num_epochs): 
        
        # take list of observations and concatenate them
        num_timesteps = 0
        # reset loss after new epoch
        loss = 0
        while num_timesteps < update_timesteps: 
            # preprocessing
            catStates = np.concatenate(states, axis=0)
            statesTorch = torch.from_numpy(catStates)/255

            action_probs = policy(statesTorch)
            m = Categorical(action_probs)
            actions = m.sample()

            rewards = [] 

            for i, (env, action) in enumerate(zip(envs, actions)):  
                # action.item()
                obs, reward, terminated, truncated, _ = env.step(action)
                num_timesteps += 1
                rewards.append(reward)

                if terminated or truncated:
                    # replace with a new actor
                    envs[i], states[i] = create_actor()
                else: 
                    # update observation
                    states[i] = np.expand_dims(obs, axis=0)

                    
            # -1 * dot product 
            loss += -1 * torch.dot(m.log_prob(actions), torch.Tensor(rewards))

        loss = loss/update_timesteps
        print("Epoch: " + str(e) + " Loss: " + str(loss) + "\n")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_model(PATH):
    env = gym.make("ALE/SpaceInvaders-ram-v5", render_mode = "human")
    obs, info = env.reset()

    # load policy network
    pol = createPolicy(all_layers)
    pol.load_state_dict(torch.load(PATH, weights_only=True))
    pol.eval()

    while True: 
        torchState = torch.from_numpy(np.expand_dims(obs, axis=0))/255

        actionNum = Categorical(pol(torchState)).sample().item()
        obs, _, terminated, truncated, _ = env.step(actionNum)

        if terminated or truncated: 
            break


'''
NUM_ACTORS = 128
pol = createPolicy(all_layers)
optimizer = optim.Adam(pol.parameters(), lr=1e-4)
train(pol, optimizer, NUM_ACTORS, 2000, 100 * NUM_ACTORS)

# also save the network
torch.save(pol.state_dict(), "model-128-100-2000")
'''


# test our network -> score was 215
# test_model("model-64-80-400")
# test_model("model-128-100-2000")





# Categorical(policy(torchState)).sample().item()
# then take action




