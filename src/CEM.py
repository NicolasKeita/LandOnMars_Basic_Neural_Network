import numpy as np
import gymnasium as gym

def policy(observation, parameters):
    # Simple linear policy: threshold angle to decide left or right
    return 0 if np.dot(parameters, observation) < 0 else 1

def run_episode(env, parameters):
    observation = env.reset()
    total_reward = 0
    for _ in range(200):  # Maximum of 200 time steps
        action = policy(observation, parameters)
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def train_cross_entropy(env, num_episodes, elite_frac, num_params):
    parameters = np.random.rand(num_params)
    num_elite = int(elite_frac * num_episodes)

    for episode in range(num_episodes):
        rewards = [run_episode(env, parameters) for _ in range(num_elite)]
        elite_indices = np.argsort(rewards)[-num_elite:]
        elite_parameters = [parameters[i] for i in elite_indices]

        # Update parameters using the elite set
        parameters = np.mean(elite_parameters, axis=0)

        # Display the average reward over the elite episodes
        print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {np.mean(rewards)}")

    return parameters

if __name__ == "__main__":
    # Create the CartPole environment
    env = gym.make('CartPole-v1')

    # Set random seed for reproducibility
    np.random.seed(42)

    # Train the Cross-Entropy Method
    num_episodes = 50
    elite_frac = 0.2
    num_params = env.observation_space.shape[0]
    learned_parameters = train_cross_entropy(env, num_episodes, elite_frac, num_params)

    # Test the learned parameters
    total_reward = run_episode(env, learned_parameters)
    print(f"Total Reward with Learned Parameters: {total_reward}")

    # Close the environment
    env.close()
