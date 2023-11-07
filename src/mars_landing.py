import math
from matplotlib import pyplot as plt, cm
from src.hyperparameters import GRAVITY


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
    norm_dist = min(1 - dist / 500, 1)
    norm_rotation = 1 - abs(rotation) / 90
    norm_vs = None
    if abs(vs) < 37:
        norm_vs = 1
    elif abs(vs) > 100:
        norm_vs = 0
    else:
        norm_vs = 1 - ((100 - abs(vs)) / (100 - 37))
    # print(norm_vs, vs)
    # norm_vs = 1 if abs(vs) < 37 else min(1 - abs(vs) / 100 - 37, 0)
    norm_hs = 1 if abs(hs) < 17 else min(1 - abs(hs) / 17, 0)

    print("CRASH x=", rocket_pos_x, ' dist=', dist, ' rot=', rotation, vs, hs, remaining_fuel, 3 * norm_vs)
    return 1 * norm_dist + 1 * norm_rotation + 3 * norm_vs + 1 * norm_hs


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
        return remaining_fuel, True
    if (rocket_pos_y < 0 or rocket_pos_y > 3000 or rocket_pos_x < 0 or rocket_pos_x > 7000
            or env[rocket_pos_y][rocket_pos_x] is False or remaining_fuel < -4):
        return normalize_unsuccessful_rewards(state), True
    return 0, False


def compute_next_state(state, action: tuple[int, int]):
    radians = action[1] * (math.pi / 180)
    x_acceleration = math.sin(radians) * action[0]
    y_acceleration = math.cos(radians) * action[0] - GRAVITY
    new_horizontal_speed = state[2] - x_acceleration
    new_vertical_speed = state[3] + y_acceleration
    new_x = state[0] + new_horizontal_speed - x_acceleration * 0.5
    new_y = state[1] + new_vertical_speed + y_acceleration * 0.5 + GRAVITY
    remaining_fuel = state[4] - action[0]
    new_state = (new_x, new_y, new_horizontal_speed,
                 new_vertical_speed, remaining_fuel, action[1],
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
        # plt.pause(0.001)
        immediate_reward, is_terminal_state = reward_function(state)
        state_value += immediate_reward
        if is_terminal_state:
            break
    return state_value
