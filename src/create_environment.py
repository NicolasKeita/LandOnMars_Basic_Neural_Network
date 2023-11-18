import math
import random

import numpy as np
from itertools import product

from src.Point2D import Point2D
from src.hyperparameters import limit_actions, GRAVITY, actions_min_max


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
        self.feature_amount = 7
        self.action_space_n = (90 + 90) * 4
        rot = range(-90, 91)
        thrust = range(5)
        self.action_space = list(product(rot, thrust))
        self.action_space_sample = lambda: random.randint(0, self.action_space_n - 1)
        self.observation_space_shape = 7000 * 3000
        self.initial_state = initial_state
        self.state = np.array(initial_state)
        self.landing_spot = landing_spot
        self.grid = grid

    def reset(self):
        self.state = np.array(self.initial_state)
        return self.state

    def step(self, action_index: int):
        action = self.action_space[action_index]
        next_state = compute_next_state(self.state, action)
        self.state = next_state
        reward, done = reward_function(next_state, self.grid, self.landing_spot)
        return next_state, reward, done, None

    def generate_random_action(self, old_rota: int, old_power_thrust: int) -> tuple[int, tuple[int, int]]:
        action_min_max = actions_min_max((old_power_thrust, old_rota))
        random_action = (
            random.randint(action_min_max[1][0], action_min_max[1][1]),
            random.randint(action_min_max[0][0], action_min_max[0][1])
        )
        return self.action_space.index(random_action), random_action

    def action_indexes_to_real_action(self, action_indexes):
        real_actions = []
        for i in action_indexes:
            real_actions.append(self.action_space[i])
        return real_actions

    def real_actions_to_indexes(self, policy):
        indexes = []
        for action in policy:
            act_1 = np.clip(round(action[0]), -90, 90)
            act_2 = np.clip(round(action[1]), 0, 4)
            indexes.append(self.action_space.index((act_1, act_2)))
        return indexes

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


def reward_function(state, grid, landing_spot) -> (float, bool):
    rocket_pos_x = round(state[0])
    rocket_pos_y = round(state[1])
    hs = state[2]
    vs = state[3]
    remaining_fuel = state[4]
    rotation = state[5]

    if (landing_spot[0].x <= rocket_pos_x <= landing_spot[1].x and landing_spot[0].y >= rocket_pos_y and
            rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20):
        print("GOOD", rocket_pos_x, remaining_fuel)
        exit(42)
        return remaining_fuel * 10, True
    if (rocket_pos_y < 0 or rocket_pos_y >= 3000 or rocket_pos_x < 0 or rocket_pos_x >= 7000
            or grid[rocket_pos_y][rocket_pos_x] is False or remaining_fuel < -4):
        return normalize_unsuccessful_rewards(state, landing_spot), True
    return 0, False


def normalize_unsuccessful_rewards(state, landing_spot):
    rocket_pos_x = round(state[0])
    hs = state[2]
    vs = state[3]
    rotation = state[5]
    dist = get_landing_spot_distance(rocket_pos_x, landing_spot[0].x, landing_spot[1].x)
    # print(dist)
    norm_dist = 1.0 if dist == 0 else max(0, 1 - dist / 7000)
    # return norm_dist
    # return norm_dist
    norm_rotation = 1 - abs(rotation) / 90
    norm_vs = 1.0 if abs(vs) <= 0 else 0.0 if abs(vs) > 120 else 1.0 if abs(vs) <= 37 else 1.0 - (abs(vs) - 37) / (120 - 37)
    norm_hs = 1.0 if abs(hs) <= 0 else 0.0 if abs(hs) > 120 else 1.0 if abs(hs) <= 17 else 1.0 - (abs(hs) - 17) / (120 - 17)
    # print("crash", dist)
    print(
        "CRASH x=", rocket_pos_x, 'dist=', dist, 'rot=', rotation, vs, hs,
          "norms:", "vs", norm_vs, "hs", norm_hs, "rotation", norm_rotation, "dist", norm_dist, "sum", (1 * norm_dist + 1 * norm_vs + 1 * norm_hs) ** 2
    )
    # return 1 * norm_dist + 1 * norm_rotation + 1 * norm_vs + 1 * norm_hs
    if dist != 0:
        return 1 * norm_dist
    # return norm_dist
    if norm_vs != 1 and norm_hs != 1:
        return (1 * norm_dist + 1 * norm_vs + 1 * norm_hs)
    # if rotation != 0:
    #     return (1 * norm_dist + 1 * norm_rotation)
    return (1 * norm_dist + 1 * norm_rotation + 1 * norm_vs + 1 * norm_hs)


# TODO no need to land in the middle
def get_landing_spot_distance(x, landing_spot_left_x, landing_spot_right_x):
    return 0 if landing_spot_left_x <= x <= landing_spot_right_x else min(abs(x - landing_spot_left_x),
                                                                          abs(x - landing_spot_right_x))
    # middle_of_landing_spot = (landing_spot_right_x + landing_spot_left_x) / 2
    # return abs(x - middle_of_landing_spot)
