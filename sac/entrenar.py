import gymnasium as gym
import time
from stable_baselines3 import PPO, SAC
import torch


env = gym.make("CarRacing-v3", continuous=True)  

#SAC
sac_model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    device="cuda",
    buffer_size=30_000,  
    batch_size=256,      
    learning_rate=3e-4,  
    tau=0.005,           
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(
        net_arch=[512, 256], 
    activation_fn=torch.nn.Tanh
    ),
    tensorboard_log="sac/logs/"
)

#SAC Training
start_time = time.time()

#Training by phases
print("FASE 1: Exploración inicial (30k steps)")
sac_model.learn(total_timesteps=30_000, tb_log_name="SAC_phase1")

print("FASE 2: Aprendizaje estable (50k steps)")  
sac_model.learning_rate = 5e-5  #Reduced LR
sac_model.learn(total_timesteps=50_000, tb_log_name="SAC_phase2")

print("FASE 3: Fine-tuning (30k steps)")
sac_model.learning_rate = 1e-5
sac_model.learn(total_timesteps=30_000, tb_log_name="SAC_phase3")


sac_time = time.time() - start_time
print(f"SAC trained in  {sac_time:.2f} seconds")

sac_model.save("sac/models/sac_model")

#SAC Evaluation
def evaluate(model, env, episodes=5):
    rewards = []
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return sum(rewards)/len(rewards)

sac_score = evaluate(sac_model, env)

print(f"Average SAC: {sac_score}")




#tensorboard --logdir=sac/logs/

#SAC entrenado en 1158.34 segundos
#SAC promedio: -17.71480848681579