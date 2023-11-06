import math
import random

import numpy as np
from matplotlib import pyplot as plt, cm
from src.hyperparameters import GRAVITY


#initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
def reward_function(state):
    rocket_pos_x = state[0]
    rocket_pos_y = state[1]
    remaining_fuel = state[4]
    env = state[7]
    landing_spot = state[8]
    if landing_spot[0].x <= rocket_pos_x <= landing_spot[1].x and landing_spot[0].y >= rocket_pos_y:  # TODO fix accuracy for reward
        return remaining_fuel
    if (rocket_pos_y < 0 or rocket_pos_y > 3000 or rocket_pos_x < 0 or rocket_pos_x > 7000
            or env[rocket_pos_y][rocket_pos_x] is False or remaining_fuel < -4):
        return -1
    return 0


def compute_next_state(state, action: tuple[int, int]):
    radians = action[1] * (math.pi / 180)
    x_acceleration = math.sin(radians) * action[0]
    y_acceleration = math.cos(radians) * action[0] - GRAVITY
    new_horizontal_speed = state[2] - x_acceleration
    new_vertical_speed = state[3] + y_acceleration
    new_x = state[0] + new_horizontal_speed - x_acceleration * 0.5
    new_y = state[1] + new_vertical_speed + y_acceleration * 0.5 + GRAVITY
    remaining_fuel = state[4] - action[0]
    new_state = (round(new_x), round(new_y), new_horizontal_speed,
                 new_vertical_speed,remaining_fuel, action[1],
                 action[0], state[7], state[8])
    return new_state


# TODO namespace mars_landing
# AKA state-value function V_pi(s)
def fitness_function(state: tuple, dna: list[tuple[int, int]], generation_id: int) -> int:
    state_value = 0
    cmap = cm.get_cmap('Set1')
    color = cmap(generation_id)
    scatter = plt.scatter(2500, 2700, color=color, label='Rocket')
    for gene in dna:
        state = compute_next_state(state, gene)
        scatter.set_offsets([state[0], state[1]])
        plt.pause(0.001)
        state_value = reward_function(state)
        if state_value != 0:
            break
    return state_value
