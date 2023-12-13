#!/usr/bin/env python
# from src.DQN.DQN import DQN_1
# if True:
#     from src.DQN.create_environment import RocketLandingEnv
# else:
#     from src.PPO_to_remove.create_environment import RocketLandingEnv
from src.GA.GeneticAlgorithm import GeneticAlgorithm
from src.GA.create_environment import RocketLandingEnv
from src.PPO_to_remove.eval_policy import eval_policy
from src.PPO_to_remove.my_PPO import PPO




if __name__ == '__main__':
    env = RocketLandingEnv()

    my_GA = GeneticAlgorithm(env)
    my_GA.learn(80 * 1000000)

    # my_DQN = DQN_1(env)
    # my_DQN.learn()
    # my_DQN.evaluate_policy()
    # my_proximal_policy_optimization = PPO(env)
    # my_proximal_policy_optimization.learn()
