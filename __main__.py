#!/usr/bin/env python

from src.my_PPO import PPO
from src.create_environment import RocketLandingEnv


if __name__ == '__main__':
    env = RocketLandingEnv()

    my_proximal_policy_optimization = PPO(env)
    my_proximal_policy_optimization.learn(10_000_000)
