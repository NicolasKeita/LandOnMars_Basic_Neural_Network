import torch

from src.PolicyGrad.network import FeedForwardNN
from src.create_environment import RocketLandingEnv


class PPO:
    def __init__(self, env: RocketLandingEnv):
        self.env = env
        self.obs_dim = 7
        self.action_dim = 2
        self.actor = FeedForwardNN(self.obs_dim, self.action_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.time_steps_per_batch = 4800  # timesteps per batch
        self.max_time_steps_per_episode = 1600  # timesteps per episode

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_time_steps=200_000_000):
        t_so_far = 0

        while t_so_far < total_time_steps:  # TODO for loop ?
            t_so_far += 1
            pass
        pass

    def get_action(self, state):  # TODO rename in something better
        return (90, 3), 5

    def rollouts(self):
        # Batch data
        batch_obs = []  # batch observations
        batch_actions = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rewards = []  # batch rewards
        batch_rewards_to_go = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        t = 0

        while t < self.time_steps_per_batch:

            ep_rewards = []
            state = self.env.reset()
            done = False

            for ep_t in range(self.max_time_steps_per_episode):
                t += 1

                # Collect observation
                batch_obs.append(state)

                # action = self.env.generate_random_action(0, 0)[1]
                action, log_prob = self.get_action(state)
                obs, rew, done, _ = self.env.step(action)

                ep_rewards.append(rew)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

        pass
