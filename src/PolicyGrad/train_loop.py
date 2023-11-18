for i in range(n_policy_updates):

    # use current policy to collect trajectories
    states, actions, weights, rewards = self._collect_trajectories(n_samples=batch_size)

    # one step of gradient ascent to update policy parameters
    loss = self._update_parameters(states, actions, weights)

    # log epoch metrics
    print('epoch: %3d \t loss: %.3f \t reward: %.3f' % (i, loss, np.mean(rewards)))
    if logger is not None:
        # we use total_steps instead of epoch to render all plots in Tensorboard comparable
        # Agents wit different batch_size (aka steps_per_epoch) are fairly compared this way.
        total_steps += batch_size
        logger.add_scalar('train/loss', loss, total_steps)
        logger.add_scalar('train/episode_reward', np.mean(rewards), total_steps)

    # evaluate the agent on a fixed set of 100 episodes
    if (i + 1) % freq_eval_in_epochs == 0:
        rewards, success = self.evaluate(n_episodes=100)

        avg_reward = np.mean(rewards)
        avg_success_rate = np.mean(success)
        if save_model and (avg_reward > best_avg_reward):
            self.save_to_disk(model_path)
            print(f'Best model! Average reward = {avg_reward:.2f}, Success rate = {avg_success_rate:.2%}')

            best_avg_reward = avg_reward