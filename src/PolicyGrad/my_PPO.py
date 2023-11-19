import torch

from src.PolicyGrad.network import FeedForwardNN
from src.create_environment import RocketLandingEnv
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(self, env: RocketLandingEnv):
        self.env = env
        self.obs_dim = 7
        self.action_dim = 2
        self.actor = FeedForwardNN(self.obs_dim, self.action_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)

        self.time_steps_per_batch = 4800  # timesteps per batch
        self.max_time_steps_per_episode = 1600  # timesteps per episode

        self.covariance_var = torch.full(size=(self.action_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.covariance_mat = torch.diag(self.covariance_var)

        self.gamma_reward_to_go = 0.95

    def learn(self, total_time_steps=200_000_000):
        t_so_far = 0

        while t_so_far < total_time_steps:  # TODO for loop ?
            t_so_far += 1
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout_batch()
            break
        pass

    def compute_rtgs(self, batch_rewards):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rewards_to_go = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rewards):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma_reward_to_go
                batch_rewards_to_go.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
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

        while t < self.time_steps_per_batch:

            ep_rewards = []
            state = self.env.reset()
            done = False

            for ep_t in range(self.max_time_steps_per_episode):  # TODO rename it horizon
                t += 1

                # Collect observation
                batch_obs.append(state)

                # action = self.env.generate_random_action(0, 0)[1]
                action, log_prob = self.get_action(state)
                obs, rew, done, _ = self.env.step(action)

                ep_rewards.append(rew)
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                break
                if done:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rewards.append(ep_rewards)
            break

        # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        # ALG STEP #4
        batch_rewards_to_go = self.compute_rtgs(batch_rewards)
        # Return the batch data
        return batch_obs, batch_actions, batch_log_probs, batch_rewards_to_go, batch_lens
