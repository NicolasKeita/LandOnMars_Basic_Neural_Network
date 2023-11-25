import math
import random
import numpy as np
from shapely import LineString

from src.hyperparameters import limit_actions, GRAVITY, actions_min_max
from src.math_utils import distance_squared_to_closest_point_to_line_segments


class RocketLandingEnv:
    def __init__(self, initial_state: list[int], landing_spot, surface: np.ndarray):
        self.feature_amount = len(initial_state)
        self.initial_state = initial_state
        self.state = np.array(initial_state)
        self.landing_spot = landing_spot
        self.surface_points = surface

        self.raw_intervals = [
            [0, 7000],  # x
            [0, 3000],  # y
            [-500, 500],  # vs
            [-500, 500],  # hs
            [-10, 20000],  # fuel remaining
            [-90, 90],  # rot
            [0, 4],  # thrust
            [0, 10_000 ** 2],  # distance squared landing_spot
            [0, 10_000 ** 2]  # distance squared surface
        ]
        self.action_constraints = [15, 1]

    @staticmethod
    def normalize_state(raw_state, raw_intervals):
        normalized_state = [(val - interval[0]) / (interval[1] - interval[0])
                            for val, interval in
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

    @staticmethod
    def normalize_action(action):
        def inv_sig(x):
            epsilon = 1e-10
            x = np.clip(x, epsilon, 1 - epsilon)
            return np.log(x / (1 - x))

        norm_dim1 = np.tanh(action[0] / 90)
        norm_dim2 = inv_sig(action[1] / 4.0)
        normalized_output = np.array([norm_dim1, norm_dim2])
        return normalized_output

    def get_action_constraints(self, previous_action):
        if previous_action is None:
            return [self.normalize_action([-15, 0]), self.normalize_action([15, 1])]
        action = self.denormalize_action(previous_action)
        legal_min_max = actions_min_max(action)
        minimun = self.normalize_action((legal_min_max[0][0], legal_min_max[1][0]))
        maximun = self.normalize_action((legal_min_max[0][1], legal_min_max[1][1]))
        return [minimun, maximun]

    def reset(self):
        self.state = np.array(self.initial_state)
        return self.normalize_state(self.state, self.raw_intervals)

    def step(self, action: tuple[int, int]):
        # action = self.action_space[action_index]
        next_state = self.compute_next_state(self.state, action)
        self.state = next_state
        reward, done = reward_function(next_state, self.landing_spot)
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

    def compute_next_state(self, state, action: tuple[int, int]):
        curr_pos = [state[0], state[1]]
        rot, thrust = limit_actions(state[5], state[6], action)
        radians = rot * (math.pi / 180)
        x_acceleration = math.sin(radians) * thrust
        y_acceleration = (math.cos(radians) * thrust) - GRAVITY
        new_horizontal_speed = state[2] - x_acceleration
        new_vertical_speed = state[3] + y_acceleration
        new_x = curr_pos[0] + state[2] - 0.5 * x_acceleration
        new_y = curr_pos[1] + state[3] + 0.5 * y_acceleration
        new_pos: LineString | list = [new_x, new_y]

        line1 = LineString(self.surface_points)
        line2 = LineString([curr_pos, new_pos])
        intersection: LineString = line1.intersection(line2)
        if intersection:
            new_pos = np.array(intersection.xy).flatten()
        remaining_fuel = state[4] - thrust

        new_state = (new_pos[0], new_pos[1], new_horizontal_speed,
                     new_vertical_speed, remaining_fuel, rot,
                     thrust,
                     distance_squared_to_closest_point_to_line_segments(np.array(new_pos), self.landing_spot),
                     distance_squared_to_closest_point_to_line_segments(new_pos, self.surface_points)
        )
        return new_state


def norm_reward(feature, interval_low, interval_high) -> float:
    return max(0.0, 1.0 - abs(feature) / interval_high)


def compute_reward(state) -> float:
    x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot_squared, dist_surface = state
    dist_normalized = norm_reward(dist_landing_spot_squared, 0, 49000000) * 2.0
    hs_normalized = norm_reward(hs, 0, 550) * 3.0
    vs_normalized = norm_reward(vs, 0, 550) * 3.0
    rotation_normalized = norm_reward(rotation, 0, 90)
    return dist_normalized + hs_normalized + vs_normalized + rotation_normalized


def reward_function(state, landing_spot) -> (float, bool):
    x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface = state

    print(state)
    is_successful_landing = (dist_landing_spot < 1 and rotation == 0 and
                             abs(vs) <= 40 and abs(hs) <= 20)
    # is_successful_landing = (landing_spot[0][0] <= x <= landing_spot[1][0] and
    #                          landing_spot[0][1] >= y and rotation == 0 and
    #                          abs(vs) <= 40 and abs(hs) <= 20)
    is_crashed = (y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or
                  dist_surface < 1 or remaining_fuel < -4)

    if is_successful_landing:
        print("SUCCESS", x, remaining_fuel)
        exit(0)
    reward = compute_reward(state)
    done = False
    if is_crashed:
        # print("CRASH", state)
        done = True
        reward -= 100
    return reward, done
