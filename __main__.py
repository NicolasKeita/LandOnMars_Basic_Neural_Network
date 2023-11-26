#!/usr/bin/env python
import numpy as np
from shapely import LineString, MultiPoint, Point

from src.my_PPO import PPO
from src.create_environment import RocketLandingEnv


def parse_planet_surface() -> MultiPoint:
    input_file = '''
        6
        0 100
        1000 500
        1500 100
        3000 100
        5000 1500
        6999 1000
    '''
    points_coordinates = np.fromstring(input_file, sep='\n', dtype=int)[1:].reshape(-1, 2)
    return MultiPoint(points_coordinates)


def find_landing_spot(planet_surface: MultiPoint) -> LineString:
    points: list[Point] = planet_surface.geoms

    for i in range(len(points) - 1):
        if points[i].y == points[i + 1].y:
            return LineString([points[i], points[i + 1]])
    raise Exception('no landing site on test-case data')


if __name__ == '__main__':
    planet_surface = parse_planet_surface()
    landing_spot = find_landing_spot(planet_surface)

    landing_spot = np.array(landing_spot.xy).T  # TODO remove
    planet_surface = np.array([(point.x, point.y) for point in planet_surface.geoms])  # TODO remove

    env = RocketLandingEnv(landing_spot, planet_surface)

    my_proximal_policy_optimization = PPO(env)
    my_proximal_policy_optimization.learn(10_000_000)
