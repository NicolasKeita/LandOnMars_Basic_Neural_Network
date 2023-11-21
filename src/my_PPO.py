import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

from src.eval_policy import eval_policy
from src.graph_handler import display_graph
from src.network import FeedForwardNN
from src.create_environment import RocketLandingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, env: RocketLandingEnv):
        self.env = env
        self.obs_dim = 7
        self.action_dim = 2

        self.time_steps_per_batch = 4800  # timesteps per batch
        self.max_time_steps_per_episode = 1600  # timesteps per episode
        self.gamma_reward_to_go = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = 0.005
        self.init_hyperparameters()

        self.actor = FeedForwardNN(self.obs_dim, self.action_dim).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = FeedForwardNN(self.obs_dim, 1).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.covariance_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(device)
        # Create the covariance matrix
        self.covariance_mat = torch.diag(self.covariance_var).to(device)

        self.roll_i = 0

    def init_hyperparameters(self):
        self.time_steps_per_batch = 4800  # timesteps per batch
        self.max_time_steps_per_episode = 1600  # timesteps per episode
        self.gamma_reward_to_go = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = 0.1

    def learn(self, total_time_steps=200_000_000):
        t_so_far = 0

        while t_so_far < total_time_steps:  # TODO for loop ?
            (batch_obs, batch_actions, batch_log_probs,
             batch_rewards_to_go, batch_lens) = self.rollout_batch()
            # print(batch_obs)
            # tmp = self.env.denormalize_state(batch_obs.numpy(), self.env.raw_intervals)
            # print(tmp)
            # exit(9)
            # display_graph(tmp, t_so_far)

            t_so_far += np.sum(batch_lens)
            print('BATCH DONE', t_so_far)

            # Calculate V_{phi, k}
            V, _ = self.evaluate(batch_obs, batch_actions)
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rewards_to_go - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_actions)
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)

                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)
                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()

                self.critic_optim.step()

        eval_policy(self.actor, self.env)
        # torch.save(self.actor.state_dict(), './model.txt')

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()
        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.covariance_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        # print("reward")
        # print(batch_rewards)
        batch_rewards_to_go = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rewards):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma_reward_to_go
                batch_rewards_to_go.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        # print(batch_rewards_to_go)
        batch_rewards_to_go = torch.tensor(batch_rewards_to_go, dtype=torch.float)
        return batch_rewards_to_go

    def get_action(self, state):  # TODO rename in something better
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(state)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.covariance_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach()

    def rollout_batch(self):
        # Batch data
        batch_obs = []  # batch observations
        batch_actions = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rewards = []  # batch rewards
        batch_rewards_to_go = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        t = 0

        self.roll_i += 1
        tmp = 0
        trajectories = []

        while t < self.time_steps_per_batch:
            tmp += 1

            ep_rewards = []
            state = self.env.reset()
            done = False

            trajectory_plot = []

            ep_t = 0
            for ep_t in range(self.max_time_steps_per_episode):  # TODO rename it horizon
                t += 1

                # Collect observation
                # print(" HERE", state)
                # print("DENORMAL", self.env.denormalize_state(state, self.env.raw_intervals))
                # print("ADDING : ", state)
                # print("ADDING 2 : ", self.env.denormalize_state(state, self.env.raw_intervals))
                # denormalized_state = self.env.denormalize_state(state, self.env.raw_intervals)
                # batch_obs.append(denormalized_state)
                batch_obs.append(state)
                # print(batch_obs)

                # action = self.env.generate_random_action(0, 0)[1]
                action, log_prob = self.get_action(state)
                action_denormalized = self.env.denormalize_action(action)
                # print(action_denormalized)
                state, reward, done, _ = self.env.step(action_denormalized)
                # print(self.env.denormalize_state(state, self.env.raw_intervals))
                trajectory_plot.append(self.env.denormalize_state(state, self.env.raw_intervals))

                ep_rewards.append(reward)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

            trajectories.append(trajectory_plot)
            # print("here", trajectory_plot)

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rewards)

        # Reshape data as tensors in the shape specified before returning
        # print(self.env.denormalize_state(batch_obs, self.env.raw_intervals))
        # print('"SEEEEEEEE')
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
        # ALG STEP #4
        batch_rewards_to_go = self.compute_rtgs(batch_rewards)

        display_graph(trajectories, self.roll_i)

        # Return the batch data
        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens
