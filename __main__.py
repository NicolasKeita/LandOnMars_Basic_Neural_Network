#!/usr/bin/env python
import numpy as np

from src.my_PPO import PPO
from src.create_environment import create_env, distance_to_line_segment, RocketLandingEnv, \
    point_to_line_distance


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

    x_max = 7000
    y_max = 3000
    grid: list[list[bool]] = create_env(planet_surface, x_max, y_max)
    landing_spot = find_landing_spot(planet_surface)
    landing_spot_points = []
    for x in range(landing_spot[0][0], landing_spot[1][0]):
        landing_spot_points.append(np.array([x, landing_spot[0][1]]))
    # initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
    # initial_state = (2500, 2700, 0, 0, 550, 0, 0)
    initial_state = np.array([
        2500, 2500, 0, 0, 500, 0, 0,
        distance_to_line_segment(np.array([500, 2700]), landing_spot_points),
        point_to_line_distance(np.array([500, 2700]), planet_surface)
    ])
    # initial_state = np.concatenate([initial_state, mars_surface.flatten()])
    # create_graph(mars_surface, 'Landing on Mars')
    env = RocketLandingEnv(initial_state, landing_spot, grid, planet_surface, landing_spot_points)

    np.set_printoptions(suppress=True)
    # torch.autograd.set_detect_anomaly(True)

    my_proximal_policy_optimization = PPO(env)
    my_proximal_policy_optimization.learn(1000_000)
    print("----------- Learn Weight ends success")
