import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

#Load all the rewards
all_rewards = []

for csv_file in glob.glob("logs/env_0/monitor.csv"):
    df = pd.read_csv(csv_file, comment='#') 
    all_rewards.append(df['r'].values)

#Calculate the mean for each 10 episodes
all_rewards_flat = np.concatenate(all_rewards)
moving_avg = pd.Series(all_rewards_flat).rolling(10).mean()

#Plot
plt.figure(figsize=(10,5))
plt.plot(all_rewards_flat, alpha=0.3, label='Reward per episode')
plt.plot(moving_avg, color='red', label='Mean by each 10 episodes')
plt.xlabel('Episode')
plt.ylabel('Accumulated reward')
plt.title('Reward evolution during training')
plt.legend()
plt.show()
