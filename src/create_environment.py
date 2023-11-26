import math
import numpy as np
from shapely import LineString, Point

from src.hyperparameters import limit_actions, GRAVITY, actions_min_max
from src.math_utils import distance_squared_to_closest_point_to_line_segments


class RocketLandingEnv:
    def __init__(self, landing_spot, surface: np.ndarray):

        initial_state = [
            2500,  # x
            2500,  # y
            0,  # horizontal speed
            0,  # vertical speed
            500,  # fuel remaining
            0,  # rotation
            0,  # thrust power
            distance_squared_to_closest_point_to_line_segments([500, 2700], landing_spot),  # distance to landing spot
            distance_squared_to_closest_point_to_line_segments([500, 2700], surface)  # distance to surface
        ]
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
        minimum = self.normalize_action((legal_min_max[0][0], legal_min_max[1][0]))
        maximum = self.normalize_action((legal_min_max[0][1], legal_min_max[1][1]))
        return [minimum, maximum]

    def reset(self):
        self.state = np.array(self.initial_state)
        return self.normalize_state(self.state, self.raw_intervals)

    def step(self, action: tuple[int, int]):
        next_state = self.compute_next_state(self.state, action)
        self.state = next_state
        reward, done = reward_function(next_state)
        next_state = self.normalize_state(next_state, self.raw_intervals)
        return next_state, reward, done, None

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
        new_pos: Point | list = [np.clip(new_x, 0, 7000), np.clip(new_y, 0, 3000)]

        line1 = LineString(self.surface_points)
        line2 = LineString([curr_pos, new_pos])
        intersection: Point = line1.intersection(line2)
        if not intersection.is_empty and isinstance(intersection, Point):
            new_pos = np.array(intersection.xy).flatten()
        remaining_fuel = state[4] - thrust

        new_state = (new_pos[0],
                     new_pos[1],
                     new_horizontal_speed,
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
    dist_normalized = norm_reward(dist_landing_spot_squared, 0, 49000000) * 8.0
    hs_normalized = norm_reward(hs, 0, 550) * 3.0
    vs_normalized = norm_reward(vs, 0, 550) * 3.0
    rotation_normalized = norm_reward(rotation, 0, 90)
    return dist_normalized + hs_normalized + vs_normalized + rotation_normalized


def reward_function(state) -> (float, bool):
    x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface = state

    is_successful_landing = (dist_landing_spot < 1 and rotation == 0 and
                             abs(vs) <= 40 and abs(hs) <= 20)
    is_crashed = (y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or
                  dist_surface < 1 or remaining_fuel < -4)

    reward = compute_reward(state)
    done = False
    if is_crashed:
        print("Crash, ", state)
        done = True
        reward -= 20
    elif is_successful_landing:
        print("SUCCESSFUL LANDING !")
        done = True
        reward += 20
        exit(0)
    return reward, done
