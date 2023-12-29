from source_code.GeneticAlgorithm import GeneticAlgorithm
from source_code.create_environment import RocketLandingEnv


if __name__ == '__main__':
    env = RocketLandingEnv()

    my_GA = GeneticAlgorithm(env)
    try:
        my_GA.learn(500)
    except KeyboardInterrupt:
        pass
