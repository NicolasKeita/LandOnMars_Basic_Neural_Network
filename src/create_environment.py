import math
import random

import numpy as np
from itertools import product

import torch
from matplotlib import pyplot as plt


from src.hyperparameters import limit_actions, GRAVITY, actions_min_max


# False = underneath the surface
def create_env(surface_points: list, x_max: int, y_max: int) -> list[list[bool]]:
    def surface_function(x, sorted_points):
        for i in range(len(sorted_points) - 1):
            x1, y1 = sorted_points[i][0], sorted_points[i][1]
            x2, y2 = sorted_points[i + 1][0], sorted_points[i + 1][1]
            if x1 <= x <= x2:
                return round(y1 + (x - x1) * (y2 - y1) / (x2 - x1))
        return 0

    world = [[False] * x_max for _ in range(y_max)]
    sorted_points = sorted(surface_points, key=lambda p: p[0])

    for x in range(x_max):
        for y in range(surface_function(x, sorted_points), y_max):
            world[y][x] = True
    return world


def display_grid(grid):
    # Convert the boolean values to integers (0 for False, 1 for True)
    array_data = np.array(grid, dtype=int)

    # Plot the binary image
    plt.imshow(array_data, cmap='binary', interpolation='none', origin='lower')
    plt.show()


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

        self.raw_intervals = [
            [0, 7000],  # x
            [0, 3000],  # y
            [-500, 500],  # vs
            [-500, 500],  # hs
            [-10, 20000],  # fuel remaining
            [-90, 90],  # rot
            [0, 4]  # thrust
        ]

    @staticmethod
    def normalize_state(raw_state, raw_intervals):
        normalized_state = [(val - interval[0]) / (interval[1] - interval[0]) for val, interval in
                            zip(raw_state, raw_intervals)]
        return np.array(normalized_state)

    @staticmethod
    def denormalize_state(normalized_state, raw_intervals):
        denormalized_state = [val * (interval[1] - interval[0]) + interval[0]
                              for val, interval in
                              zip(normalized_state, raw_intervals)]
        return np.array(denormalized_state)

    @staticmethod
    def denormalize_action(raw_output):
        def sig(x):
            return 1 / (1 + np.exp(-np.clip(x, -700, 700)))
        output_dim1 = np.round(np.tanh(raw_output[0]) * 90.0)
        output_dim2 = np.round(sig(raw_output[1]) * 4.0)
        output = np.array([output_dim1, output_dim2], dtype=int)
        return output

    def reset(self):
        self.state = np.array(self.initial_state)
        return self.normalize_state(self.state, self.raw_intervals)

    def step(self, action: tuple[int, int]):
        # action = self.action_space[action_index]
        next_state = compute_next_state(self.state, action)
        self.state = next_state
        reward, done = reward_function(next_state, self.grid, self.landing_spot)
        next_state = self.normalize_state(next_state, self.raw_intervals)
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
    # print("action_raw:", action, state[5], state[6])
    rot, thrust = limit_actions(state[5], state[6], action)
    # print("action_limited", action)
    # state = np.round(state).astype(int)
    # print("Compute next_state_1: ", state, rot, thrust)
    radians = rot * (math.pi / 180)
    x_acceleration = math.sin(radians) * thrust
    y_acceleration = (math.cos(radians) * thrust) - GRAVITY
    new_horizontal_speed = state[2] - x_acceleration
    new_vertical_speed = state[3] + y_acceleration
    new_x = state[0] + state[2] - 0.5 * x_acceleration
    # new_x = state[1] + (((state[3] ** 2) + (new_vertical_speed ** 2)) / (2 * y_acceleration))
    # print("suvat =", state[1] + (((state[3] ** 2) + (new_vertical_speed ** 2)) / (2 * y_acceleration)))

    new_y = state[1] + state[3] + 0.5 * y_acceleration
    # new_y = state[1] + (((state[3] ** 2) + (new_vertical_speed ** 2)) / (2 * y_acceleration))
    remaining_fuel = state[4] - thrust
    new_state = (new_x, new_y, new_horizontal_speed,
                 new_vertical_speed, remaining_fuel, rot,
                 thrust)
    # print("NEW STATE: ", new_state)
    # print("ROUNDED", np.round(new_state))
    return new_state


def compute_reward(state, landing_spot) -> float:
    x1 = landing_spot[0].x
    y1 = landing_spot[0].y
    x2 = landing_spot[1].x
    y2 = landing_spot[1].y
    x0 = state[0]
    y0 = state[1]
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    distance = numerator / denominator
    return -distance


def reward_function(state, grid, landing_spot) -> (float, bool):
    x, y, hs, vs, remaining_fuel, rotation, thrust = state

    is_successful_landing = (landing_spot[0][0] <= x <= landing_spot[1][0] and
                             landing_spot[0][1] >= y and rotation == 0 and
                             abs(vs) <= 40 and abs(hs) <= 20)
    # print(state)

    is_crashed = (y < 0 or y >= 3000-1 or x < 0 or x >= 7000-1 or
                  grid[round(y)][round(x)] is False or remaining_fuel < -4)

    if is_successful_landing:
        print("GOOD", x, remaining_fuel)
        exit(42)
        # return remaining_fuel * 10, True
    elif is_crashed:
        return normalize_unsuccessful_rewards(state, landing_spot), True
    else:
        return 0, False
        # return compute_reward(state, landing_spot), False


def normalize_unsuccessful_rewards(state, landing_spot):
    rocket_pos_x = round(state[0])
    hs = state[2]
    vs = state[3]
    rotation = state[5]
    dist = get_landing_spot_distance(rocket_pos_x, landing_spot[0][0], landing_spot[1][0])
    # print(dist)
    norm_dist = 1.0 if dist == 0 else max(0, 1 - dist / 7000)
    return norm_dist
    # return norm_dist
    norm_rotation = 1 - abs(rotation) / 90
    norm_vs = 1.0 if abs(vs) <= 0 else 0.0 if abs(vs) > 120 else 1.0 if abs(vs) <= 37 else 1.0 - (abs(vs) - 37) / (
                120 - 37)
    norm_hs = 1.0 if abs(hs) <= 0 else 0.0 if abs(hs) > 120 else 1.0 if abs(hs) <= 17 else 1.0 - (abs(hs) - 17) / (
                120 - 17)
    # print("crash", dist)
    # print(
    #     "CRASH x=", rocket_pos_x, 'dist=', dist, 'rot=', rotation, vs, hs,
    #     "norms:", "vs", norm_vs, "hs", norm_hs, "rotation", norm_rotation, "dist", norm_dist, "sum",
    #     (1 * norm_dist + 1 * norm_vs + 1 * norm_hs)
    # )
    # return 1 * norm_dist + 1 * norm_rotation + 1 * norm_vs + 1 * norm_hs

    # print("Crash ! , ", norm_dist, norm_vs, norm_hs, norm_dist + norm_vs + norm_hs)
    return (1 * norm_dist + 1 * norm_vs + 1 * norm_hs)
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
