import gymnasium
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv

model_class = DDPG  # works also with SAC, DDPG and TD3
N_BITS = 15

# import numpy as np

# a = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1])
# if isinstance(a, int):
#     exit(1)
# exit(2)


env = BitFlippingEnv(n_bits=N_BITS, continuous=True, max_steps=N_BITS)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
    ),
    verbose=2,
)

# Train the model
model.learn(10_000)

model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = model_class.load("./her_bit_env", env=env)

obs, info = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    print(obs)
    if terminated or truncated:
        obs, info = env.reset()
