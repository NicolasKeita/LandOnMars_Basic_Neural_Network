import numpy as np

from src.GA.GeneticAlgorithm import GeneticAlgorithm
from src.GA.create_environment import RocketLandingEnv


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    env = RocketLandingEnv()

    my_GA = GeneticAlgorithm(env)
    try:
        my_GA.learn(8300)
    except KeyboardInterrupt:
        pass
