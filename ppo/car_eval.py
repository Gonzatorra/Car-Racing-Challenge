import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import imageio

def make_env():
    return gym.make("CarRacing-v3", render_mode="rgb_array")

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

model = PPO.load("model/ppo_car_racing_v3")

frames = []
obs = env.reset()
total_reward = 0
terminated, truncated = False, False

while not (terminated or truncated):
    frame = env.envs[0].render()
    frames.append(frame)

    action, _ = model.predict(obs)
    obs, rewards, dones, infos = env.step(action)
    total_reward += rewards[0]

    terminated, truncated = dones[0], infos[0].get("TimeLimit.truncated", False)

print(f"Reward: {total_reward:.2f}")
imageio.mimsave("result/car_racing_v3_result.mp4", frames, fps=30)
env.close()

