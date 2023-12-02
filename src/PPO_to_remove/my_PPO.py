import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

from src.graph_handler import display_graph, plot_mean_rewards, create_graph, plot_terminal_state_rewards
from src.network import FeedForwardNN
from src.create_environment import RocketLandingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, env: RocketLandingEnv):
        self.env = env

        self.obs_dim = 6
        self.action_dim = self.env.action_space_dimension

        self.time_steps_per_batch = 1 + 80 * 0  # TODO increase
        self.max_time_steps_per_episode = 700
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        # self.lr = 0.0005
        self.lr = 0.07

        if False:
            self.actor = FeedForwardNN(self.obs_dim, self.action_dim, device).to(device)
            self.actor.load_state_dict(torch.load('actor.pt'))
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
            self.actor.eval()
            self.critic = FeedForwardNN(self.obs_dim, 1, device).to(device)
            self.critic.load_state_dict(torch.load('critic.pt'))
            self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
            self.critic.eval()
        else:
            self.actor = FeedForwardNN(self.obs_dim, self.action_dim, device).to(device)
            self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
            self.critic = FeedForwardNN(self.obs_dim, 1, device).to(device)
            self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.covariance_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(device)
        self.covariance_mat = torch.diag(self.covariance_var).to(device)

        self.roll_i = 0
        self.max_grad_norm = 0.5

        self.entropy_coefficient = 0.2
        self.target_kl = 0.02
        self.lam = 0.98
        self.gamma_reward_to_go = 0.95

        fig, (ax_terminal_state_rewards, ax_mean_rewards, ax_trajectories) = plt.subplots(3,
                                                                                          1, figsize=(10, 8))
        self.fig = fig
        self.ax_rewards = ax_mean_rewards
        self.ax_trajectories = ax_trajectories
        self.ax_terminal_state_rewards = ax_terminal_state_rewards
        create_graph(self.env.surface, 'Landing on Mars', ax_trajectories)

    def learn(self, total_time_steps=200_000_000):
        torch.autograd.set_detect_anomaly(True)
        t_so_far = 0
        i_so_far = 0
        mean_rewards_history = []
        terminal_state_reward_history = []

        while t_so_far < total_time_steps:  # TODO for loop ?
            (batch_obs, batch_actions, batch_log_probs,
             batch_rewards, batch_lens, batch_vals,
             batch_dones, batch_rewards_not_normalized) = self.rollout_batch()
            A_k = self.calculate_gae(batch_rewards, batch_vals, batch_dones, True)

            V: torch.Tensor = self.critic(batch_obs).squeeze(-1)
            batch_rewards_to_go = A_k + V.detach()

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            for i in range(self.n_updates_per_iteration):
                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_actions)

                log_ratios = curr_log_probs - batch_log_probs
                ratios = torch.exp(log_ratios)
                approx_kl = ((ratios - 1) - log_ratios).mean()

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                entropy_loss = entropy.mean()

                actor_loss = actor_loss - self.entropy_coefficient * entropy_loss
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                critic_loss: torch.Tensor = nn.MSELoss()(V, batch_rewards_to_go)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()


                print(f"Iteration {i_so_far}")
                print(f"Actor Loss: {actor_loss.item()}")
                print(f"Critic Loss: {critic_loss.item()}")
                print(f"Entropy Loss: {entropy_loss.item()}")
                print(f"Approx KL Divergence: {approx_kl.item()}")

                if approx_kl > self.target_kl:
                    break

            avg_reward = np.mean(np.concatenate(batch_rewards))
            mean_rewards_history.append(avg_reward)

            avg_reward = []
            for batch in batch_rewards:
                avg_reward.append(batch[-1])
            avg_reward = np.mean(avg_reward)
            terminal_state_reward_history.append(avg_reward)

            plot_mean_rewards(mean_rewards_history, ax=self.ax_rewards)
            plot_terminal_state_rewards(terminal_state_reward_history, ax=self.ax_terminal_state_rewards)

            torch.save(self.actor.state_dict(), './actor.pt')
            torch.save(self.critic.state_dict(), './critic.pt')

    def evaluate(self, batch_obs, batch_acts):
        V: torch.Tensor = self.critic(batch_obs).squeeze(-1)
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.covariance_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs, dist.entropy()

    def get_action(self, state, action_constraints):
        mean = self.actor(state)
        dist = MultivariateNormal(mean, self.covariance_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def rollout_batch(self):
        batch_obs = []  # batch observations
        batch_actions = []  # batch actions
        # batch_log_probs = []  # log probs of each action
        batch_log_probs = []
        batch_rewards = []  # batch rewards
        batch_lens = []  # episodic lengths in batch
        batch_vals = []
        batch_dones = []
        batch_rewards_not_normalized = []

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

                state_features = self.env.extract_features(state)
                batch_obs.append(state_features)

                action_constraints = self.env.get_action_constraints(prev_action)
                action_tensor, log_prob = self.get_action(state_features, action_constraints)
                action = action_tensor.tolist()
                prev_action = action
                val = self.critic(state_features).detach()

                state, reward, done, _ = self.env.step(self.env.denormalize_action(action))
                ep_dones.append(done)
                trajectory_plot.append(self.env.denormalize_state(state))

                ep_rewards.append(reward)
                ep_vals.append(val)
                batch_actions.append(action_tensor)
                batch_log_probs.append(log_prob)
                if done:
                    break

            trajectories.append(trajectory_plot)

            batch_lens.append(ep_t + 1)
            # ep_rewards_normalized = min_max_scaling(ep_rewards)
            ep_rewards_normalized = ep_rewards

            batch_rewards.append(ep_rewards_normalized)
            ep_vals = torch.stack(ep_vals).squeeze(-1)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            batch_rewards_not_normalized.append(ep_rewards)

        batch_obs = torch.FloatTensor(batch_obs).to(device)
        batch_actions = torch.stack(batch_actions)
        batch_vals = torch.stack(batch_vals)
        batch_log_probs = torch.stack(batch_log_probs).detach()

        display_graph(trajectories, self.roll_i, ax=self.ax_trajectories)

        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens, batch_vals, batch_dones, batch_rewards_not_normalized

    def calculate_gae(self, rewards, values: torch.Tensor, dones, normalize = False) -> torch.Tensor:
        advantages = []
        advantage = 0
        v_next = 0

        for i in range(len(rewards)):
            for r, v in zip(reversed(rewards[i]), reversed(values[i])):
                td_error = r + v_next * self.gamma_reward_to_go - v
                advantage = td_error + advantage * self.gamma_reward_to_go * self.lam
                v_next = v

                advantages.insert(0, advantage)
            print("rewards:", rewards)
            print("rewards mean:", np.mean(rewards))
            advantages = torch.stack(advantages)
        if normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages


def min_max_scaling(ep_rewards):
    max_reward = 5

    # x_min, x_max = 0, max_reward
    return [reward / max_reward for reward in ep_rewards]
    # x_min_last, x_max_last = -10, max_reward + 10
    # return [(reward - x_min_last) / (x_max_last - x_min_last)
    #         if i == len(ep_rewards) - 1
    #         else (reward - x_min) / (x_max - x_min)
    #         for i, reward in enumerate(ep_rewards)]
