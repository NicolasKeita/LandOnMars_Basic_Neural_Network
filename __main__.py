#!/usr/bin/env python
import numpy as np

from src.math_utils import distance_squared_to_closest_point_to_line_segments
from src.my_PPO import PPO
from src.create_environment import RocketLandingEnv


def parse_planet_surface() -> np.ndarray:
    input_file = '''
        6
        0 100
        1000 500
        1500 100
        3000 100
        5000 1500
        6999 1000
    '''
    return np.fromstring(input_file, sep='\n', dtype=int)[1:].reshape(-1, 2)

#TODO move to env
def find_landing_spot(planet_surface):
    for i in range(len(planet_surface) - 1):
        if planet_surface[i + 1][1] == planet_surface[i][1]:
            landing_spot_start = (planet_surface[i][0], planet_surface[i][1])
            landing_spot_end = (planet_surface[i + 1][0], planet_surface[i + 1][1])
            return landing_spot_start, landing_spot_end
    raise Exception('no landing site on test-case data')


if __name__ == '__main__':
    planet_surface = parse_planet_surface()
    landing_spot = find_landing_spot(planet_surface)

    env = RocketLandingEnv(landing_spot, planet_surface)

    my_proximal_policy_optimization = PPO(env)
    my_proximal_policy_optimization.learn(10_000_000)
