# Define a simple neural network for the policy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.create_environment import RocketLandingEnv
from src.graph_handler import display_graph


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


n_episodes = 1000

input_dim = 7
output_dim = 720
policy = PolicyNetwork(7, 720)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)


def collect_trajectories(policy, env, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        states, actions, rewards = [], [], []
        state = env.reset()
        tmp = []
        while True:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state)

            # Get action probabilities from the policy
            action_probs = policy(state_tensor)

            # Sample an action
            action = torch.multinomial(action_probs, 1).item()

            # Take the action in the environment
            tmp.append(env.action_space[action])
            next_state, reward, done, _ = env.step(action)

            # Save the state, action, and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break

        # Append the trajectory
        trajectories.append((states, actions, rewards))

    return trajectories


def compute_discounted_rewards(rewards, gamma=0.99):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


# Function to train the policy using Vanilla Policy Gradient
def train_policy(policy, optimizer, trajectories):
    policy_loss: int | torch.Tensor = 0

    for states, actions, rewards in trajectories:
        # Convert states to PyTorch tensor
        states_tensor = torch.FloatTensor(states)

        # Convert rewards to PyTorch tensor
        rewards_tensor = torch.FloatTensor(compute_discounted_rewards(rewards))

        # Get action probabilities from the policy
        action_probs = policy(states_tensor)

        # Calculate the log probabilities of the taken actions
        selected_action_probs = action_probs.gather(1, torch.LongTensor(actions).view(-1, 1))

        # Compute the policy loss (negative log-likelihood)
        loss = -torch.sum(torch.log(selected_action_probs) * rewards_tensor)

        # Accumulate the policy loss
        policy_loss += loss

    # Backpropagation
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


def eval_loop(env: RocketLandingEnv):
    for episode in range(n_episodes):

        trajectories = collect_trajectories(policy, env, num_trajectories=1)
        display_graph(trajectories[0][0], episode)
        train_policy(policy, optimizer, trajectories)
        total_reward = sum(sum(t[2]) for t in trajectories)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")



    # state = env.reset()
    # solution = []
    # while True:
    #     trajectories = collect_trajectories(policy, env, num_trajectories=1)
    #
    #     # Sample an action from the learned policy
    #     action = np.argmax(probs)
    #
    #     solution.append(action)
    #
    #     # Take the action in the environment
    #     next_state, _, done, _ = env.step(action)
    #
    #     state = next_state
    #
    #     if done:
    #         print(solution)
    #         print(env.action_indexes_to_real_action(solution))
    #         break