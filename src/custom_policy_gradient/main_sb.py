import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

# Create the environment
env = gym.make('MountainCarContinuous-v0', render_mode='human')

# Instantiate the PPO1 model
model = PPO(MlpPolicy, env, verbose=1)

# Train the model
model.learn(total_timesteps=10_000)

# Evaluate the model
obs, _ = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, _, info = env.step(action)
    env.render()
    if done:
        print('Final Position:', env.get_state()[0])
        break
