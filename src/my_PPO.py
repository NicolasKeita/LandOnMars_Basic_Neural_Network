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

        self.obs_dim = self.env.feature_amount
        self.action_dim = 2

        self.time_steps_per_batch = 1 + 80 * 10  # timesteps per batch
        self.max_time_steps_per_episode = 100  # timesteps per episode
        self.gamma_reward_to_go = 0.95
        self.n_updates_per_iteration = 10
        self.clip = 0.2 # As recommended by the paper
        self.lr = 0.001

        self.actor = FeedForwardNN(self.obs_dim, self.action_dim, device).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = FeedForwardNN(self.obs_dim, 1, device).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.covariance_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(device)
        self.covariance_mat = torch.diag(self.covariance_var).to(device)

        self.roll_i = 0

        self.reward_mean = 0
        self.reward_std = 1
        self.max_grad_norm = 0.5

        self.ent_coef = 5000
        self.target_kl = 0.02

        self.lam = 0.98
        self.gamma = self.gamma_reward_to_go

    def learn(self, total_time_steps=200_000_000):
        t_so_far = 0
        i_so_far = 0

        while t_so_far < total_time_steps:  # TODO for loop ?
            (batch_obs, batch_actions, batch_log_probs,
             batch_rewards, batch_lens, batch_vals, batch_dones) = self.rollout_batch()
            A_k = self.calculate_gae(batch_rewards, batch_vals, batch_dones)
            V = self.critic(batch_obs).squeeze()
            batch_rewards_to_go = A_k + V.detach()

            t_so_far += np.sum(batch_lens)
            print('BATCH DONE', t_so_far)
            i_so_far += 1

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_actions)

            # A_k = batch_rewards_to_go - V.detach()
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                frac = (t_so_far - 1.0) / total_time_steps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0)
                self.actor_optim.param_groups[0]["lr"] = new_lr
                self.critic_optim.param_groups[0]["lr"] = new_lr


                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_actions)
                logratios = curr_log_probs - batch_log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()


                surr1 = ratios * A_k
                # surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                surr2 = ratios * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                # Entropy Regularization
                entropy_loss = entropy.mean()
                # Discount entropy loss by given coefficient
                actor_loss = actor_loss - self.ent_coef * entropy_loss

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                critic_loss = nn.MSELoss()(V, batch_rewards_to_go)
                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()

                if approx_kl > self.target_kl:
                    break # if kl aboves threshold

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
        return V, log_probs, dist.entropy()

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

    def get_action(self, state, action_constraints):  # TODO rename fct name to something better
        state = torch.tensor(state, dtype=torch.float).to(device)
        mean = self.actor(state)
        dist = MultivariateNormal(mean, self.covariance_mat)
        action = dist.sample().cpu().numpy()
        action[0] = np.clip(action[0], action_constraints[0][0], action_constraints[1][0])
        action[1] = np.clip(action[1], action_constraints[0][1], action_constraints[1][1])
        log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float).to(device))

        # action[0] = action[]
        # constrained_action = prev_action + np.clip(new_action - prev_action, -0.15, 0.15)
        # log_prob = dist.log_prob(action)

        # return action.detach().cpu().numpy(), log_prob.detach()
        return action, log_prob.detach()

    def rollout_batch(self):
        batch_obs = []  # batch observations
        batch_actions = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rewards = []  # batch rewards
        batch_lens = []  # episodic lengths in batch
        batch_vals = []
        batch_dones = []

        t = 0
        self.roll_i += 1
        trajectories = []

        while t < self.time_steps_per_batch:

            ep_rewards = []
            ep_vals = []
            state = self.env.reset()
            prev_action = None

            ep_dones = []
            done = False

            trajectory_plot = []

            ep_t = 0
            for ep_t in range(self.max_time_steps_per_episode):  # TODO rename it horizon
                t += 1
                ep_dones.append(done)
                batch_obs.append(state)

                action_constraints = self.env.get_action_constraints(prev_action)
                action, log_prob = self.get_action(state, action_constraints)
                prev_action = action
                val = self.critic(state)

                state, reward, done, _ = self.env.step(self.env.denormalize_action(action))
                state = torch.tensor(state, dtype=torch.float).to(device)
                trajectory_plot.append(self.env.denormalize_state(state, self.env.raw_intervals))

                ep_rewards.append(reward)
                ep_vals.append(val.flatten())
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                if done:
                    break

            trajectories.append(trajectory_plot)

            batch_lens.append(ep_t + 1)
            ep_rewards = z_score_normalization(ep_rewards)
            batch_rewards.append(ep_rewards)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)

        # batch_rewards = torch.tensor(batch_rewards, dtype=torch.float).to(device)
        # batch_vals = torch.tensor(batch_vals, dtype=torch.float).to(device)
        # batch_dones = torch.tensor(batch_dones, dtype=torch.float).to(device)

        display_graph(trajectories, self.roll_i)

        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens, batch_vals, batch_dones

    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            ep_dones = torch.tensor(ep_dones, dtype=torch.float).to(device)
            for t in reversed(range(len(ep_rews))):
                # if t + 1 < len(ep_rews):
                #     delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                if t + 1 < len(ep_rews) and not ep_dones[t + 1]:
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            # batch_advantages.extend(advantages)

            advantages = torch.tensor(advantages, dtype=torch.float).to(device)
            # advantages = torch.clamp(advantages, -0.2, 0.2)  # Adjust the clipping range as needed
            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).to(device)

def z_score_normalization(ep_rewards):
    mean = np.mean(ep_rewards)
    std_dev = np.std(ep_rewards)
    return [(reward - mean) / (std_dev + 1e-8) for reward in ep_rewards]


