import gym
import numpy as np

env = gym.make("Pendulum-v1", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   # action = policy(observation)  # User-defined policy function
   action = np.array([2.0])
   print(observation)

   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()
env.close()