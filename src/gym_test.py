import math
import random
from pprint import pprint

# Define the environment (toy example)
num_states = 10
num_actions = 2


# Initialize the policy: A simple softmax policy
def initialize_policy(num_states, num_actions):
    policy = []
    for _ in range(num_states):
        action_probabilities = [random.random() for _ in range(num_actions)]
        total_prob = sum(action_probabilities)
        normalized_probabilities = [p / total_prob for p in action_probabilities]
        policy.append(normalized_probabilities)
    return policy


# Initialize the policy
policy = initialize_policy(num_states, num_actions)
#
# [[0.6538727147317213, 0.3461272852682788],
#  [0.7107829434280045, 0.28921705657199537],
#  [0.727003690652844, 0.272996309347156],
#  [0.23619036977784488, 0.7638096302221551],
#  [0.2219136384583975, 0.7780863615416025],
#  [0.9934262386717752, 0.0065737613282248605],
#  [0.30798047916915583, 0.6920195208308442],
#  [0.8269548762671406, 0.17304512373285943],
#  [0.32621715141188884, 0.6737828485881112],
#  [0.4040063804533317, 0.5959936195466682]]
#

# Hyperparameters
learning_rate = 0.01
num_episodes = 1000

# Training loop
for episode in range(num_episodes):
    state = random.randint(0, num_states - 1)
    episode_states = []
    episode_actions = []
    episode_rewards = []

    # Generate an episode
    for _ in range(100):  # Cap episode length for safety
        # Sample an action from the policy
        action = random.choices(range(num_actions), weights=policy[state])[0]

        # Store state, action, and reward
        episode_states.append(state)
        episode_actions.append(action)

        # Reward function (in a real environment, this would be defined)
        if state == num_states - 1:
            reward = 1
        else:
            reward = 0

        episode_rewards.append(reward)

        # Transition to the next state
        state = random.randint(0, num_states - 1)

    # Update the policy at the end of the episode using REINFORCE
    total_return = sum(episode_rewards)
    for t in range(len(episode_states)):
        state_t = episode_states[t]
        action_t = episode_actions[t]

        # Compute the return from time step t
        G_t = sum(episode_rewards[t:])

        # Update the policy using the REINFORCE update rule
        policy[state_t][action_t] += learning_rate * G_t

# Print the learned policy
for state, probabilities in enumerate(policy):
    print(f"State {state}: Action Probabilities {probabilities}")
