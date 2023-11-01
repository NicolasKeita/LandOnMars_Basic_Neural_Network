# Define the grid world environment
import random

from src.Rocket import State

grid_world = [
    [0, 0, 0, 0],
    [0, 1, 0, 1],
    [0, 0, 0, 0]
]


# Define the feature extractor function
def extract_features(state):
    # For each state, use four features: x-coordinate, y-coordinate, wall indicator, and a constant feature.
    x, y = state
    wall_indicator = grid_world[y][x]
    return [x, y, wall_indicator, 1]


# Initialize the weights for the linear function approximation
weights = [0.0, 0.0, 0.0, 0.0]


# Define the policy (e.g., epsilon-greedy)
def policy(state, epsilon):
    if random.random() < epsilon:
        # Exploration: Choose a random action
        return random.choice([0, 1, 2, 3])
    else:
        # Exploitation: Choose the action with the highest estimated value
        features = extract_features(state)
        values = [sum(w * f for w, f in zip(weights, features)) for _ in range(4)]
        print(values)
        return values.index(max(values))


# Simulate episodes using Monte Carlo method
num_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1


def take_action(state, action):
    pass


for episode in range(num_episodes):
    # Initialize the episode
    episode_states = []
    episode_rewards = []

    state = (0, 0)
    while True:
        action = policy(state, epsilon)
        next_state, reward = take_action(state, action)  # Define take_action() based on the environment
        episode_states.append(state)
        episode_rewards.append(reward)
        state = next_state

        if state == (3, 2):
            # Reach the goal, end the episode
            break

    # Update the state-value function using Monte Carlo returns
    G = 0
    for t in range(len(episode_states) - 1, -1, -1):
        G = gamma * G + episode_rewards[t]
        features = extract_features(episode_states[t])
        for i in range(4):
            weights[i] += alpha * (G - sum(w * f for w, f in zip(weights, features))) * features[i]

# Now you can use the learned weights to make predictions for new states.
new_state = (1, 2)
new_features = extract_features(new_state)
estimated_value = sum(w * f for w, f in zip(weights, new_features))
print(f"Estimated value for new state {new_state}: {estimated_value}")
