#!/usr/bin/env python
import numpy as np

from src.my_PPO import PPO
from src.create_environment import RocketLandingEnv, distance_squared_to_closest_point_to_line_segments


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


def find_landing_spot(mars_surface):
    for i in range(len(mars_surface) - 1):
        if mars_surface[i + 1][1] == mars_surface[i][1]:
            landing_spot_start = (int(mars_surface[i][0]), int(mars_surface[i][1]))
            landing_spot_end = (int(mars_surface[i + 1][0]), int(mars_surface[i + 1][1]))
            return np.array([landing_spot_start, landing_spot_end])
    raise Exception('no landing site on test-case data')


if __name__ == '__main__':
    planet_surface = parse_planet_surface()
    landing_spot = find_landing_spot(planet_surface)
    initial_state = np.array([
        2500,  # x
        2500,  # y
        0,  # horizontal speed
        0,  # vertical speed
        500,  # fuel remaining
        0,  # rotation
        0,  # thrust power
        distance_squared_to_closest_point_to_line_segments(np.array([500, 2700]), landing_spot),  # distance to landing spot
        distance_squared_to_closest_point_to_line_segments(np.array([500, 2700]), planet_surface)  # distance to surface
    ])
    # initial_state = np.concatenate([initial_state, mars_surface.flatten()])
    # create_graph(mars_surface, 'Landing on Mars')
    env = RocketLandingEnv(initial_state, landing_spot, planet_surface)

    np.set_printoptions(suppress=True)
    # torch.autograd.set_detect_anomaly(True)

    my_proximal_policy_optimization = PPO(env)
    my_proximal_policy_optimization.learn(1000_000)
    print("----------- Learn Weight ends success")
