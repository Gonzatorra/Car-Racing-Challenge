import gymnasium as gym
from stable_baselines3 import SAC

# Crear el entorno
env = gym.make("CarRacing-v3", continuous=True, render_mode="human")

# Cargar el modelo entrenado
model_path = "sac/sac_model.zip"
model = SAC.load(model_path, env=env, device="cuda")

# Función para ejecutar varios episodios
def run_episodes(model, env, num_episodes=5):
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episodio {ep+1}: Recompensa total = {total_reward}")

# Ejecutar episodios
run_episodes(model, env, num_episodes=5)

# Cerrar el entorno al finalizar
env.close()
