import optuna
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

def evaluate(model, env, n_eval_episodes=5):
    total_rewards = []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, truncated = False, False
        ep_reward = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
        total_rewards.append(ep_reward)
    return sum(total_rewards) / len(total_rewards)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    env = DummyVecEnv([lambda: gym.make("CarRacing-v3")])
    env = VecFrameStack(env, n_stack=4)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
    )

    model.learn(total_timesteps=200_000)
    mean_reward = evaluate(model, env)
    env.close()
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15)
print("Mejores:", study.best_params)
