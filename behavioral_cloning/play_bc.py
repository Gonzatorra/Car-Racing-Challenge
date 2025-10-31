import gymnasium as gym
import torch
from train_bc import CNNPolicyClone, action_dim
import numpy as np

#Create the environment so we can visualize it
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)

#Load the trained model
model = CNNPolicyClone(action_dim)
model.load_state_dict(torch.load("models/bc_model.pth"))
model.eval()

obs, _ = env.reset()

while True:
    #Normalize the observation
    obs_input = torch.tensor(obs/255.0, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        #Get the result, the vector that the network predicts --> [steering, gas, brake]
        action = model(obs_input).squeeze(0).numpy()
    #Put that action into the environment
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
