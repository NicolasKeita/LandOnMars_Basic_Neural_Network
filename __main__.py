#!/usr/bin/env python
from src.DQN.DQN import DQN_1
from src.DQN.create_environment import RocketLandingEnv
from src.PPO_to_remove.eval_policy import eval_policy
from src.PPO_to_remove.my_PPO import PPO


if __name__ == '__main__':
    env = RocketLandingEnv()

    my_DQN = DQN_1(env)
    my_DQN.learn()
