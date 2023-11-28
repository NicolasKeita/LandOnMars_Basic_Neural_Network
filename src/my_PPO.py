import torch
from matplotlib import pyplot as plt
from shapely import MultiPoint
from torch import nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import numpy as np

from src.eval_policy import eval_policy
from src.graph_handler import display_graph, plot_mean_rewards, create_graph, plot_terminal_state_rewards
from src.network import FeedForwardNN
from src.create_environment import RocketLandingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(self, env: RocketLandingEnv):
        self.env = env

        self.obs_dim = len(self.env.state)
        self.action_dim = 2

        self.time_steps_per_batch = 1 + 80 * 0  # timesteps per batch
        self.max_time_steps_per_episode = 700  # timesteps per episode
        self.gamma_reward_to_go = 0.98
        # self.gamma_reward_to_go = 0.80
        # self.gamma_reward_to_go = 1
        self.n_updates_per_iteration = 9
        self.clip = 0.2
        # self.lr = 0.01
        self.lr = 0.001
        self.actor = FeedForwardNN(self.obs_dim, self.action_dim, device).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic = FeedForwardNN(self.obs_dim, 1, device).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.covariance_var = torch.full(size=(self.action_dim,), fill_value=0.000000000000001).to(device)
        # self.covariance_var = torch.full(size=(self.action_dim,), fill_value=0.5).to(device)
        self.covariance_mat = torch.diag(self.covariance_var).to(device)

        self.roll_i = 0

        self.max_grad_norm = 0.5

        self.ent_coef = 0
        self.target_kl = 0.02
        self.lam = 0.98
        self.gamma = self.gamma_reward_to_go

        fig, (ax_terminal_state_rewards, ax_mean_rewards, ax_trajectories) = plt.subplots(3,
                                                                                          1, figsize=(10, 8))
        self.fig = fig
        self.ax_rewards = ax_mean_rewards
        self.ax_trajectories = ax_trajectories
        self.ax_terminal_state_rewards = ax_terminal_state_rewards
        create_graph(self.env.surface, 'Landing on Mars', ax_trajectories)

    def learn(self, total_time_steps=200_000_000):
        t_so_far = 0
        i_so_far = 0
        mean_rewards_history = []
        terminal_state_reward_history = []

        while t_so_far < total_time_steps:  # TODO for loop ?
            (batch_obs, batch_actions, batch_log_probs,
             batch_rewards, batch_lens, batch_vals, batch_dones, batch_rewards_not_normalized) = self.rollout_batch()
            A_k = self.calculate_gae(batch_rewards, batch_vals, batch_dones)
            V: torch.Tensor = self.critic(batch_obs).squeeze()
            batch_rewards_to_go = torch.add(A_k, V.detach())

            t_so_far += np.sum(batch_lens)
            print('BATCH DONE (more like steps)', t_so_far)
            print("Iter so far ", i_so_far)
            i_so_far += 1

            # print(A_k)
            # A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            # print(A_k)
            # exit(9)

            for i in range(self.n_updates_per_iteration):
                # frac = (t_so_far - 1.0) / total_time_steps
                # new_lr = self.lr * (1.0 - frac)
                # new_lr = max(new_lr, 0)
                # print(new_lr)

                # self.actor_optim.param_groups[0]["lr"] = new_lr
                # self.critic_optim.param_groups[0]["lr"] = new_lr

                V, curr_log_probs, entropy = self.evaluate(batch_obs, batch_actions)

                log_ratios = curr_log_probs - batch_log_probs[0]  # TODO remove [0]
                ratios = torch.exp(log_ratios)
                # print(ratios)
                approx_kl = ((ratios - 1) - log_ratios).mean()

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                # surr2 = ratios * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()

                entropy_loss = entropy.mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optim.step()

                critic_loss: torch.Tensor = nn.MSELoss()(V, batch_rewards_to_go)
                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
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

        eval_policy(self.actor, self.env)
        # torch.save(self.actor.state_dict(), './model.txt')

    # TODO check if I need to constraints this too
    def evaluate(self, batch_obs, batch_acts) -> tuple[torch.TensorType, any, any]:
        V = self.critic(batch_obs).squeeze()
        mean = self.actor(batch_obs)
        # for i, batch_ob in enumerate(batch_obs):
        #     ob = batch_ob.cpu().detach().numpy()
        #     previous_rot = ob[5]
        #     previous_thrust = ob[6]
        #
        #     previous_action = self.env.denormalize_action([previous_rot, previous_thrust])
        #     action_constraints = self.env.get_action_constraints(previous_action)
        #     n_mean = torch.clamp_(mean[i],
        #                           torch.tensor(action_constraints[0], dtype=torch.float, device=device),
        #                           torch.tensor(action_constraints[1], dtype=torch.float, device=device))
        mean[:, 0] = 0
        dist = MultivariateNormal(mean, self.covariance_mat)
        log_probs = dist.log_prob(batch_acts)
        return V, log_probs, dist.entropy()

    def get_action(self, state, action_constraints):  # TODO rename fct name to something better
        mean = self.actor(state)
        n_mean = torch.clamp(mean,
                             torch.tensor(action_constraints[0], dtype=torch.float, device=device),
                             torch.tensor(action_constraints[1], dtype=torch.float, device=device))
        n_mean[0] = 0

        dist = MultivariateNormal(n_mean, self.covariance_mat)
        # print(dist)
        # exit(0)

        # action: np.ndarray = dist.sample().cpu().numpy()
        action = dist.sample()
        # action = np.clip(action, action_constraints[0], action_constraints[1])
        # action[0] = 0
        # log_prob = dist.log_prob(torch.tensor(action, dtype=torch.float).to(device))
        log_prob = dist.log_prob(action)
        return action, log_prob

    def rollout_batch(self):
        batch_obs = []  # batch observations
        batch_actions = []  # batch actions
        batch_log_probs = []  # log probs of each action
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
                ep_dones.append(done)
                batch_obs.append(state)

                action_constraints = self.env.get_action_constraints(prev_action)
                action, log_prob = self.get_action(state, action_constraints)
                action = action.cpu().detach()
                prev_action = action
                val = self.critic(state)

                state, reward, done, _ = self.env.step(self.env.denormalize_action(action))
                trajectory_plot.append(self.env.denormalize_state(state, self.env.raw_intervals))

                ep_rewards.append(reward)
                ep_vals.append(val.flatten())
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                # print(log_prob)
                if done:
                    break

            trajectories.append(trajectory_plot)

            batch_lens.append(ep_t + 1)
            ep_rewards_normalized = min_max_scaling(ep_rewards)

            batch_rewards.append(ep_rewards_normalized)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            batch_rewards_not_normalized.append(ep_rewards)

        batch_obs = np.array(batch_obs)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_actions = np.array(batch_actions)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float).to(device)
        # print(batch_log_probs)
        # exit(1)
        # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)

        display_graph(trajectories, self.roll_i, ax=self.ax_trajectories)

        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens, batch_vals, batch_dones, batch_rewards_not_normalized

    def calculate_gae(self, rewards, values, dones) -> torch.Tensor:
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

            # advantages = torch.tensor(advantages, dtype=torch.float).to(device)
            # advantages = torch.clamp(advantages, -0.2, 0.2)  # Adjust the clipping range as needed
            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).to(device)


def min_max_scaling(ep_rewards):
    sum = 4
    # sum = 10000
    x_min, x_max = 0, sum
    x_min_last, x_max_last = -10, sum + 10
    return [(reward - x_min_last) / (x_max_last - x_min_last)
            if i == len(ep_rewards) - 1
            else (reward - x_min) / (x_max - x_min)
            for i, reward in enumerate(ep_rewards)]
