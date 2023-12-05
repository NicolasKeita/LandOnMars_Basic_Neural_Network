from src.PPO_stable_baseline.create_environment import RocketLandingEnv
from stable_baselines3 import PPO

env = RocketLandingEnv()

model = PPO("MlpPolicy", env, verbose=2)
model.learn(total_timesteps=5_000)
model.save("model_ppo_stable_baseline")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
