import math
from matplotlib import pyplot as plt, cm
from src.hyperparameters import GRAVITY, action_1_min_max, action_2_min_max, limit_actions


def get_landing_spot_distance(x, landing_spot_left_x, landing_spot_right_x):
    return 0 if landing_spot_left_x <= x <= landing_spot_right_x else min(abs(x - landing_spot_left_x),
                                                                          abs(x - landing_spot_right_x))


def normalize_unsuccessful_rewards(state):
    rocket_pos_x = round(state[0])
    hs = state[2]
    vs = state[3]
    remaining_fuel = state[4]  # TODO maybe include ?
    rotation = state[5]
    landing_spot = state[8]
    dist = get_landing_spot_distance(rocket_pos_x, landing_spot[0].x, landing_spot[1].x)
    norm_dist = 1.0 if dist == 0 else max(0, 1 - dist / 7000)
    norm_rotation = 1 - abs(rotation) / 90
    norm_vs = 1.0 if abs(vs) <= 0 else 0.0 if abs(vs) > 120 else 1.0 if abs(vs) <= 37 else 1.0 - (abs(vs) - 37) / (120 - 37)
    norm_hs = 1.0 if abs(hs) <= 0 else 0.0 if abs(hs) > 120 else 1.0 if abs(hs) <= 17 else 1.0 - (abs(hs) - 17) / (120 - 17)
    print("CRASH x=", rocket_pos_x, 'dist=', dist, 'rot=', rotation, vs, hs, remaining_fuel,
          "norms:", "vs", norm_vs, "hs", norm_hs, "rotation", norm_rotation, "dist", norm_dist, "sum", 1 * norm_dist + 1 * norm_rotation + 1 * norm_vs + 1 * norm_hs)
    # return 1 * norm_dist + 1 * norm_rotation + 1 * norm_vs + 1 * norm_hs
    if dist != 0:
        return 1 * norm_dist
    if rotation != 0:
        return 1 * norm_dist + 1 * norm_rotation
    return 1 * norm_dist + 1 * norm_rotation + 1 * norm_vs + 1 * norm_hs


# initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
def reward_function(state):
    rocket_pos_x = round(state[0])
    rocket_pos_y = round(state[1])
    hs = state[2]
    vs = state[3]
    remaining_fuel = state[4]
    rotation = state[5]
    env = state[7]
    landing_spot = state[8]

    if (landing_spot[0].x <= rocket_pos_x <= landing_spot[1].x and landing_spot[0].y >= rocket_pos_y and
            rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20):
        print("GOOD", rocket_pos_x, remaining_fuel)
        return remaining_fuel * 10, True
    if (rocket_pos_y < 0 or rocket_pos_y > 3000 or rocket_pos_x < 0 or rocket_pos_x > 7000
            or env[rocket_pos_y][rocket_pos_x] is False or remaining_fuel < -4):
        return normalize_unsuccessful_rewards(state), True
    return 0, False


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
                 thrust, state[7], state[8])
    return new_state


# TODO namespace mars_landing
# AKA state-value function V_pi(s)
def fitness_function(state: tuple, dna: list[tuple[int, int]]) -> tuple[int, list[float], list[float]]:
    state_value = 0
    trajectory_x = [state[0]]
    trajectory_y = [state[1]]

    for gene in dna:
        state = compute_next_state(state, gene)
        trajectory_x.append(state[0])
        trajectory_y.append(state[1])
        immediate_reward, is_terminal_state = reward_function(state)
        state_value = immediate_reward
        if state_value > 100:
            print(dna)
            print(state)
            exit(0)
        if is_terminal_state:
            break
    return state_value, trajectory_x, trajectory_y
