import gym
import numpy as np

# Create the CartPole-v1 environment
env = gym.make('CartPole-v1')

# Define the policy neural network
def simple_policy(observation, theta):
    # Calculate the score for taking action 0 or 1
    score = np.dot(theta, observation)
    action = 1 if score > 0 else 0
    return action

# Training parameters
num_episodes = 1000
learning_rate = 0.01

# Policy parameter (theta) initialization
theta = np.random.rand(4)

for episode in range(num_episodes):
    episode_rewards = []
    episode_actions = []
    observation = env.reset()

    while True:
        # Render the environment (you can comment this out for faster training)
        env.render()

        # Sample an action from the policy
        action = simple_policy(observation, theta)
        episode_actions.append(action)

        # Take the chosen action and observe the next state and reward
        observation, reward, done, _ = env.step(action)

        # Record the reward
        episode_rewards.append(reward)

        if done:
            break

    # Calculate the cumulative rewards
    cumulative_rewards = np.cumsum(episode_rewards)

    # Update the policy using the policy gradient theorem
    for t in range(len(episode_actions)):
        G_t = sum(episode_rewards[t:])
        theta += learning_rate * (G_t - np.dot(theta, observation)) * observation

    # Print the episode information
    print(f"Episode {episode + 1}: Total Reward: {sum(episode_rewards)}")

# Close the environment
env.close()
