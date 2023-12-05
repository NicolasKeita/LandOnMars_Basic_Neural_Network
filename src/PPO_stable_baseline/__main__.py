from src.PPO_stable_baseline.create_environment import RocketLandingEnv
from stable_baselines3 import PPO

env = RocketLandingEnv()

model = PPO("MlpPolicy", env, policy_kwargs=dict(net_arch=[256, 256]),  verbose=1, use_sde=False)
model.learn(total_timesteps=5000_000)
model.save("model_ppo_stable_baseline")
#https://stable-baselines3.readthedocs.io/en/master/guide/export.html#manual-export

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    # print(action)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
