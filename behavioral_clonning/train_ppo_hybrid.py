if __name__ == "__main__":    
    import gymnasium as gym
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
    from stable_baselines3.common.monitor import Monitor
    from train_bc import CNNPolicyClone, action_dim
    import os

    #Create log folder
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)

    #Another CNN for PPO - Extract visual characteristics and 
    #states that are not in human data.
    class CustomCNNExtractor(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=512):
            super().__init__(observation_space, features_dim)
            #Same architecture than the BC CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                c, h, w = observation_space.shape
                sample = torch.zeros(1, c, h, w)
                conv_out_dim = self.cnn(sample).reshape(1, -1).size(1)

            self.linear = nn.Sequential(
                nn.Linear(conv_out_dim, features_dim),
                nn.ReLU()
            )

        def forward(self, x):
            x = x / 255.0 #Normalize pixels
            x = self.cnn(x)
            x = self.linear(x)
            return x

    #Environment with Monitor to register each episode
    def make_env(rank):
        def _init():
            env = gym.make("CarRacing-v3", render_mode=None)  # NO human
            env_log_dir = os.path.join(log_dir, f"env_{rank}")
            os.makedirs(env_log_dir, exist_ok=True)
            env = Monitor(env, env_log_dir)
            return env
        return _init

    num_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])


    #PPO
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=512)
    )

    ppo_model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_steps=1024
    )

    #Load BC weights
    bc_model = CNNPolicyClone(action_dim)
    bc_model.load_state_dict(torch.load("models/bc_model.pth", map_location="cpu"))
    ppo_model.policy.features_extractor.cnn.load_state_dict(bc_model.conv.state_dict(), strict=False)
    print("Weigths loaded in PPO")

    #Train PPO
    ppo_model.learn(total_timesteps=1_000_000)
    ppo_model.save("ppo_hybrid_model")
    print("Train fishished")
