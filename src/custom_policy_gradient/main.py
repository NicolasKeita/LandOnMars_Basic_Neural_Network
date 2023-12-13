import numpy as np
import gymnasium as gym

# Environment
env = gym.make('MountainCarContinuous-v0', render_mode='human')

# Policy parameters (proportional to state)
num_state_dimensions = env.observation_space.shape[0]
policy_weights = np.random.rand(num_state_dimensions)
policy_intercept = 0.0

# Hyperparameters
learning_rate = 0.01
discount_factor = 0.99
num_episodes = 1000

for episode in range(num_episodes):
    # Collect trajectory
    states, actions, rewards = [], [], []
    state, _ = env.reset()

    while True:
        # Calculate action proportional to state
        action = np.dot(state, policy_weights) + policy_intercept
        action = np.clip(action, env.action_space.low, env.action_space.high)

        next_state, reward, done, _, _ = env.step(action)

        # Record trajectory
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        env.render()
        if done:
            break

    # Compute discounted rewards
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * discount_factor + rewards[t]
        discounted_rewards[t] = running_add

    # Policy update (proportional to state)
    for i in range(num_state_dimensions):
        policy_weights[i] += learning_rate * np.sum(discounted_rewards * np.array(states)[:, i])
    policy_intercept += learning_rate * np.sum(discounted_rewards)

    # Print episode statistics
    print(f"Episode {episode + 1}: Total Reward: {np.sum(rewards)}")

# Close the environment
env.close()
