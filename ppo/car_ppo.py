import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

env = make_vec_env("CarRacing-v3", n_envs=1)
env = VecFrameStack(env, n_stack=4)

eval_env = make_vec_env("CarRacing-v3", n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=4)

#para parar antes si el reward es bueno
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    callback_on_new_best=stop_callback
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_car_racing_v3_tensorboard/",
    learning_rate=0.00046992234890011204,
    n_steps=4096,
    batch_size=32,
    n_epochs=10,
    gamma=0.9504846377016437,
    gae_lambda=0.9382292917166312,
    clip_range=0.14477260869966714,
)

model.learn(total_timesteps=200_000, callback=eval_callback)
model.save("ppo_car_racing_v3")
env.close()
