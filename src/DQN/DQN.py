# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
import math
import random
from itertools import count

import matplotlib
import torch
from matplotlib import pyplot as plt
from torch import optim, nn

from src.DQN.ReplayMemory import ReplayMemory, Transition
from src.DQN.create_environment import RocketLandingEnv
from src.DQN.network import DQN


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN_1:
    def __init__(self, env: RocketLandingEnv):
        self.env = env

        self.steps_done = 0
        self.policy_net = None
        self.target_net = None
        self.episode_durations = []
        self.memory = None
        self.optimizer = None

        self.reward_history = []

    def learn(self):

        # Get number of actions from gym action space
        n_actions = self.env.action_space_discrete_n
        # Get the number of state observations
        state, _ = self.env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

        if torch.cuda.is_available():
            num_episodes = 180000
        else:
            num_episodes = 50

        for i_episode in range(num_episodes):
            total_reward = 0
            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                action_denormalized = self.env.action_space[action.item()]
                observation, reward, done, _ = self.env.step(action_denormalized)
                reward = torch.tensor([reward], device=device)
                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.reward_history.append(reward.item())
                    self.plot_rewards()  # Call the new method to plot mean rewards

                    # self.episode_durations.append(t + 1)
                    # self.plot_durations()
                    break

        print('Complete')
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.show()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            state = self.env.denormalize_state(state[0])
            random_action = self.env.generate_random_action(round(state[5].item()), round(state[6].item()))
            return torch.tensor([[random_action[0]]], device=device, dtype=torch.long)

    def plot_rewards(self, show_result=False, window_size=100):
        plt.figure(1)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward History')  # Change ylabel to 'Reward History'

        plt.plot(self.reward_history, label='Episode Reward')

        # Calculate the mean over a specified window
        if len(self.reward_history) >= window_size:
            means = [sum(self.reward_history[i - window_size + 1:i + 1]) / window_size for i in
                     range(window_size - 1, len(self.reward_history))]
            plt.plot(range(window_size - 1, len(self.reward_history)), means, label=f'Mean over {window_size} episodes',
                     color='red')

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    # def plot_durations(self, show_result=False):
    #     plt.figure(1)
    #     durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    #     if show_result:
    #         plt.title('Result')
    #     else:
    #         plt.clf()
    #         plt.title('Training...')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Duration')
    #     plt.plot(durations_t.numpy())
    #     # Take 100 episode averages and plot them too
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())
    #
    #     plt.pause(0.001)  # pause a bit so that plots are updated
    #     if is_ipython:
    #         if not show_result:
    #             display.display(plt.gcf())
    #             display.clear_output(wait=True)
    #         else:
    #             display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        tmp = self.policy_net(state_batch)

        state_action_values = tmp.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()