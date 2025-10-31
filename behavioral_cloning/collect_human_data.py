import gymnasium as gym
import numpy as np
import pickle
import pygame
import os

#Create the environment
env = gym.make("CarRacing-v3", render_mode="human", continuous=True)
obs, _ = env.reset()
done = False

#Initialize action: [steering, gas, brake]
action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
human_data = []

#Initialize pygame clock for FPS control
clock = pygame.time.Clock()

print("Controls: Arrows LEFT/RIGHT = turn, UP = speed, DOWN = break")
print("Press ESC to end the episode.")

while not done:
    #Get the movements
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action[0] = -1.0
            elif event.key == pygame.K_RIGHT:
                action[0] = 1.0
            elif event.key == pygame.K_UP:
                action[1] = 1.0
            elif event.key == pygame.K_DOWN:
                action[2] = 0.8
            elif event.key == pygame.K_ESCAPE:
                done = True
        elif event.type == pygame.KEYUP:
            if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                action[0] = 0.0
            elif event.key == pygame.K_UP:
                action[1] = 0.0
            elif event.key == pygame.K_DOWN:
                action[2] = 0.0

    #Execute the actions in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    done = done or terminated or truncated

    #Save the actions (obs, action)
    human_data.append((obs.copy(), action.copy()))

    clock.tick(30)



#Save human data
os.makedirs("data", exist_ok=True)
with open("data/human_data.pkl", "wb") as f:
    pickle.dump(human_data, f)

print(f"Saved {len(human_data)} human transactions")
env.close()
