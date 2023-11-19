# Define a simple neural network for the policy
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import init

from src.create_environment import RocketLandingEnv
from src.graph_handler import display_graph


class PolicyEstimator():
    def __init__(self):
        self.num_observations = 7
        self.num_actions = 5

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation))


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, output_size)
        self.softmax = nn.Softmax(dim=-1)

        # Initialize parameters randomly
        self.init_weights()

    def init_weights(self):
        for layer in [self.fc, self.fc2]:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
                init.zeros_(layer.bias)


    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


n_episodes = 100

input_dim = 7
output_dim = 5


def collect_trajectories(estimator, env, num_trajectories):
    trajectories = []
    for _ in range(num_trajectories):
        states, actions, rewards = [], [], []
        state = env.reset()
        tmp = []
        while True:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state)

            # Get action probabilities from the policy
            # action_probs = estimator.predict(state).detach().numpy()
            # print(action_probs)
            action_probs = estimator(state_tensor)

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
    # discounted_rewards = []
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std())
    return discounted_rewards

def calculateLoss(gamma=0.99):

    # calculating discounted rewards:
    rewards = []
    dis_reward = 0
    for reward in self.rewards[::-1]:
        dis_reward = reward + gamma * dis_reward
        rewards.insert(0, dis_reward)

    # normalizing the rewards:
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std())

    loss = 0
    for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
        advantage = reward - value.item()
        action_loss = -logprob * advantage
        value_loss = F.smooth_l1_loss(value, reward)
        loss += (action_loss + value_loss)
    return loss


# Function to train the policy using Vanilla Policy Gradient
def train_policy(policy, optimizer, trajectories):
    policy_loss: int | torch.Tensor = 0
    for states, actions, rewards in trajectories:
        states_tensor = torch.FloatTensor(np.array(states))
        rewards_tensor = torch.FloatTensor(compute_discounted_rewards(rewards))
        action_probs = policy(states_tensor)
        # print(action_probs)
        # Calculate the log probabilities of the taken actions
        selected_action_probs = action_probs.gather(1, torch.LongTensor(actions).view(-1, 1))
        # print(selected_action_probs)
        # exit(9)

        # Compute the policy loss (negative log-likelihood)
        loss = -torch.sum(torch.log(selected_action_probs) * rewards_tensor)

        # Accumulate the policy loss
        policy_loss += loss

    # Backpropagation
    optimizer.zero_grad()
    # loss_good = calculateLoss()
    policy_loss.backward()
    optimizer.step()
    # optimizer.zero_grad()


def eval_loop(env: RocketLandingEnv):
    policy = PolicyNetwork(input_dim, output_dim)
    # estimator = PolicyEstimator()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    # optimizer = optim.Adam(estimator.network.parameters(), lr=0.01)

    for episode in range(n_episodes):

        trajectories = collect_trajectories(policy, env, num_trajectories=1)
        # pprint(trajectories[0])
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