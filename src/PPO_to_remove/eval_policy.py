"""
	This file is used only to evaluate our trained policy/actor after
	training in main_tmp.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without
	relying on ppo.py.
"""


def _log_summary(ep_len, ep_ret, ep_num, ep_actions):
    """
        Print to stdout what we've logged so far in the most recent episode.

        Parameters:
            None

        Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))
    list_of_tuples = [tuple(arr) for arr in ep_actions]

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(list_of_tuples, flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(policy, env, render):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on.

        Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on
            render - Specifies whether to render or not

        Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.

        Note:
            If you're unfamiliar with Python generators, check this out:
                https://wiki.python.org/moin/Generators
            If you're unfamiliar with Python "yield", check this out:
                https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    """
    # Rollout until user kills process
    print("-----------------------------------------------")
    print("-----------------------------------------------")
    print("-----------------------------------------------")
    print("-----------------------------------------------")
    for i in range(5):
        obs = env.reset()
        done = False

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return

        actions = []

        while not done:
            t += 1

            # Render environment if specified, off by default
            if render:
                env.render()

            # Query deterministic action from policy and run it
            action = policy(obs).detach().cpu().numpy()
            action = env.denormalize_action(action)

            # print('rot?', obs[5], 'power=?', obs[6])
            # obs_tmp = env.denormalize_state(obs, env.raw_intervals)
            # print('rot?', obs_tmp[5], 'power=?', obs_tmp[6])
            # action_tmp = limit_actions(obs_tmp[5], obs_tmp[6], action)
            # actions.append(env.denormalize_action(action))
            # action = (0, 1)
            actions.append(action)

            obs, rew, done, _ = env.step(action)

            # Sum all episodic rewards as we go along
            ep_ret += rew

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret, actions


def eval_policy(policy, env, render=False):
    """
        The main function to evaluate our policy with. It will iterate a generator object
        "rollout", which will simulate each episode and return the most recent episode's
        length and return. We can then log it right after. And yes, eval_policy will run
        forever until you kill the process.

        Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
            render - Whether we should render our episodes. False by default.

        Return:
            None

        NOTE: To learn more about generators, look at rollout's function description
    """
    # Rollout with the policy and environment, and log each episode's data
    for ep_num, (ep_len, ep_ret, actions) in enumerate(rollout(policy, env, render)):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num, ep_actions=actions)