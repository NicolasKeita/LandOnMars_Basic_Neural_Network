import math
import random

import numpy as np

from src.Point2D import Point2D
from src.hyperparameters import limit_actions, GRAVITY


# False = underneath the surface
def create_env(surface_points: list[Point2D], x_max: int, y_max: int) -> list[list[bool]]:
    def surface_function(x, sorted_points):
        for i in range(len(sorted_points) - 1):
            x1, y1 = sorted_points[i].x, sorted_points[i].y
            x2, y2 = sorted_points[i + 1].x, sorted_points[i + 1].y
            if x1 <= x <= x2:
                return round(y1 + (x - x1) * (y2 - y1) / (x2 - x1))
        return 0

    world = [[False] * x_max for _ in range(y_max)]
    sorted_points = sorted(surface_points, key=lambda p: p.x)

    for x in range(x_max):
        for y in range(surface_function(x, sorted_points), y_max):
            world[y][x] = True
    return world


class RocketLandingEnv:
    def __init__(self, initial_state: tuple, landing_spot, grid):
        # self.observation_space_shape = 7
        self.feature_amount = 11
        self.action_space_n = 90 + 90 * 5
        self.action_space_sample = lambda: (random.randint(0, 4), random.randint(-90, 90))
        self.initial_state = initial_state
        self.landing_spot = landing_spot
        self.state = np.array(initial_state)
        # self.landing_spot_left_x = landing_spot[0].x

    def reset(self):
        self.state = np.array(self.initial_state)
        return self.state

    def step(self, action):
        next_state = compute_next_state(self.state, action)
        self.state = next_state
        print(next_state)
        reward = None
        done = False
        return next_state, reward, done


def compute_next_state(state, action: tuple[int, int]):
    rot, thrust = limit_actions(state[5], state[6], action)
    radians = rot * (math.pi / 180)
    x_acceleration = math.sin(radians) * thrust
    y_acceleration = math.cos(radians) * thrust - GRAVITY
    new_horizontal_speed = state[2] - x_acceleration
    new_vertical_speed = state[3] + y_acceleration
    new_x = state[0] + new_horizontal_speed - x_acceleration * 0.5
    new_y = state[1] + new_vertical_speed + y_acceleration * 0.5 + GRAVITY
    remaining_fuel = state[4] - thrust
    new_state = (new_x, new_y, new_horizontal_speed,
                 new_vertical_speed, remaining_fuel, rot,
                 thrust)
    return new_state
