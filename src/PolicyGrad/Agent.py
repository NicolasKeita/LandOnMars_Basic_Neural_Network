from src.create_environment import RocketLandingEnv


class RandomAgent:

    def __init__(self, env: RocketLandingEnv):
        self.env = env

    def act(self, state) -> int:
        return self.env.generate_random_action(0, 0)[0]
