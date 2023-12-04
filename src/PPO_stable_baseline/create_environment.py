import math
import random
from itertools import product

import gymnasium
import numpy as np
from gymnasium import spaces
from shapely import LineString, Point, MultiPoint
from src.PPO_to_remove.math_utils import distance_squared_to_line


class RocketLandingEnv(gymnasium.Env):
    def __init__(self):
        surface_points = self.parse_planet_surface()
        self.surface = LineString(surface_points.geoms)
        self.landing_spot = self.find_landing_spot(surface_points)
        self.initial_state = [
            2500,  # x
            2500,  # y
            0,  # horizontal speed
            0,  # vertical speed
            10000,  # fuel remaining
            0,  # rotation
            0,  # thrust power
            distance_squared_to_line([500, 2700], self.landing_spot),  # distance to landing spot
            distance_squared_to_line([500, 2700], self.surface)  # distance to surface
        ]
        self.state_intervals = [
            [0, 7000],  # x
            [0, 3000],  # y
            [-200, 200],  # vs
            [-200, 200],  # hs
            [0, 10000],  # fuel remaining
            [-90, 90],  # rot
            [0, 4],  # thrust
            [0, 3000 ** 2],  # distance squared landing_spot
            [0, 3000 ** 2]  # distance squared surface
        ]
        self.state = self.initial_state
        self.action_constraints = [15, 1]
        self.action_space_dimension = 2
        self.action_space_discrete_n = 905
        self.gravity = 3.711
        self.initial_fuel = 10_000

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        rot = range(-90, 91)
        thrust = range(5)
        self.action_space = [list(action) for action in product(rot, thrust)]
        # self.action_space = list(product(rot, thrust))
        self.action_space_sample = lambda: random.randint(0, self.action_space_discrete_n - 1)

    def action_indexes_to_real_action(self, action_indexes: list):
        real_actions = []
        for i in action_indexes:
            real_actions.append(self.action_space[i])
        return real_actions

    def real_actions_to_indexes(self, policy: list):
        indexes = []
        for action in policy:
            act_1 = np.clip(round(action[0]), -90, 90)
            act_2 = np.clip(round(action[1]), 0, 4)
            indexes.append(self.action_space.index((act_1, act_2)))
        return indexes

    def generate_random_action(self, old_rota: int, old_power_thrust: int) -> tuple[int, list]:
        action_min_max = actions_min_max([old_rota, old_power_thrust])
        a = action_min_max[0][0]
        b = action_min_max[1][0]
        c = action_min_max[1][1]
        action_min_max = [[int(num) for num in sublist] for sublist in action_min_max]

        random_action = [
            random.randint(action_min_max[0][0], action_min_max[0][1]),
            random.randint(action_min_max[1][0], action_min_max[1][1])
        ]
        return self.action_space.index(random_action), random_action

    def extract_features(self, state):
        # Create a new array without x, y, and thrust
        return state[2:6] + state[7:]

    def sample_action(self):
        return [0, 0] # TODO

    @staticmethod
    def parse_planet_surface():
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

    @staticmethod
    def find_landing_spot(planet_surface: MultiPoint) -> LineString:
        points: list[Point] = planet_surface.geoms

        for i in range(len(points) - 1):
            if points[i].y == points[i + 1].y:
                return LineString([points[i], points[i + 1]])
        raise Exception('no landing site on test-case data')

    def normalize_state(self, raw_state: list):
        return [(val - interval[0]) / (interval[1] - interval[0])
                for val, interval in zip(raw_state, self.state_intervals)]

    def denormalize_state(self, normalized_state: list):
        return [val * (interval[1] - interval[0]) + interval[0]
                for val, interval in zip(normalized_state, self.state_intervals)]

    @staticmethod
    def denormalize_action(normalized_actions: list):
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -100, 100)))

        action_1 = np.round(np.tanh(normalized_actions[0]) * 90.0)
        action_2 = np.round(sigmoid(normalized_actions[1]) * 4.0)
        return [action_1, action_2]

    @staticmethod
    def normalize_action(raw_actions: list):
        def inv_sigmoid(x):
            epsilon = 1e-10
            x = np.clip(x, epsilon, 1 - epsilon)
            return np.log(x / (1 - x))

        action_1 = np.arctan(raw_actions[0] / 90)
        action_2 = inv_sigmoid(raw_actions[1] / 4.0)
        return [action_1, action_2]

    # TODO check if I really need this function
    def get_action_constraints(self, normalized_previous_action: list):
        if normalized_previous_action is None:
            return [self.normalize_action([-15, 0]), self.normalize_action([15, 1])]
        action = self.denormalize_action(normalized_previous_action)
        legal_min_max = actions_min_max(action)
        minimum = self.normalize_action([legal_min_max[0][0], legal_min_max[1][0]])
        maximum = self.normalize_action([legal_min_max[0][1], legal_min_max[1][1]])
        return [minimum, maximum]

    def reset(self):
        self.state = self.initial_state
        return self.normalize_state(self.state), None

    def step(self, action_to_do: list):
        self.state = self.compute_next_state(self.state, action_to_do)
        reward, done = self.reward_function(self.state)
        next_state_normalized = self.normalize_state(self.state)
        return next_state_normalized, reward, done, None

    def compute_next_state(self, state, action: list):
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot_squared, dist_surface = state
        rot, thrust = limit_actions(rotation, thrust, action)
        radians = rot * (math.pi / 180)
        x_acceleration = math.sin(radians) * thrust
        y_acceleration = (math.cos(radians) * thrust) - self.gravity
        new_horizontal_speed = hs - x_acceleration
        new_vertical_speed = vs + y_acceleration
        new_x = x + hs - 0.5 * x_acceleration
        new_y = y + vs + 0.5 * y_acceleration
        new_pos: Point | list = [np.clip(new_x, 0, 7000), np.clip(new_y, 0, 3000)]

        line1 = LineString(self.surface)
        line2 = LineString([[x, y], new_pos])
        intersection: Point = line1.intersection(line2)
        if not intersection.is_empty and isinstance(intersection, Point):
            new_pos = np.array(intersection.xy).flatten()
        remaining_fuel = max(remaining_fuel - thrust, 0)

        new_state = [new_pos[0],
                     new_pos[1],
                     new_horizontal_speed,
                     new_vertical_speed, remaining_fuel, rot,
                     thrust,
                     distance_squared_to_line(np.array(new_pos), self.landing_spot),
                     distance_squared_to_line(new_pos, self.surface)]
        return new_state

    def close(self):
        pass


    def compute_reward(self, state) -> float:
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot_squared, dist_surface = state
        speed_scaling = 1  # Scaling factor for speed rewards
        rotation_scaling = 1  # Scaling factor for rotation rewards
        dist_normalized = norm_reward(dist_landing_spot_squared, 0, 9_000_000) * 1
        dist_to_surface_normalized = norm_reward(dist_surface, 0, 9_000_000)
        is_close_to_land = dist_landing_spot_squared < 700 ** 2
        if abs(hs) <= 20:
            hs_normalized = 1 * speed_scaling * dist_to_surface_normalized
        else:
            hs_normalized = norm_reward(hs, 0, 200) * speed_scaling * dist_to_surface_normalized

        if abs(vs) <= 40:
            vs_normalized = 1
            # vs_normalized = 1 * speed_scaling * dist_to_surface_normalized
        else:
            vs_normalized = norm_reward(vs, 0, 200)
            # vs_normalized = norm_reward(vs, 0, 300) * speed_scaling * dist_to_surface_normalized
        # print(vs_normalized, vs, thrust)
        # print("here")

        # if is_close_to_land:
        #     rotation_normalized = norm_reward(rotation, 0, 90) * rotation_scaling * dist_normalized
        # else:
        #     rotation_normalized = 1
        rotation_normalized = norm_reward(rotation, 0, 90)
        fuel_normalized = norm_reward(self.initial_fuel - remaining_fuel, 0, self.initial_fuel)

        # return rotation_normalized * 20
        # return vs_normalized
        # altitude_penalty = max(0, high_altitude - optimal_altitude) / 3000
        vertical_speed_penalty = min(0, vs + 40) / -90

        # return - vertical_speed_penalty
        return dist_normalized * 2 + hs_normalized + vs_normalized + rotation_normalized + fuel_normalized

    def reward_function(self, state: list) -> tuple[float, bool]:
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface = state

        is_successful_landing = (dist_landing_spot < 1 and rotation == 0 and
                                 abs(vs) <= 40 and abs(hs) <= 20)
        is_crashed_anywhere = (y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or
                               dist_surface < 1 or remaining_fuel < -4)
        is_crashed_on_landing_spot = dist_landing_spot < 1

        reward = 0
        done = False
        if is_successful_landing:
            print("SUCCESSFUL LANDING !", state)
            done = True
            reward = 7 + norm_reward(self.initial_fuel - remaining_fuel, 0, self.initial_fuel) * 5
            return reward, done
        elif is_crashed_on_landing_spot:
            formatted_list = [f"{label}: {round(value):04d}" for label, value in
                              zip(['vs', 'hs', 'rotation'], [vs, hs, rotation])]
            print('Crash on landing side', formatted_list)
            done = True
            reward = norm_reward(abs(vs), 40, 200) * 4 + norm_reward(abs(hs), 20, 200) + norm_reward_to_the_fourth(abs(rotation), 0, 90) * 2
            # reward = norm_reward(abs(vs), 40, 200)
            # print(reward, norm_reward(abs(vs), 40, 200) * 3, norm_reward(abs(hs), 20, 200) * 1 / 2, norm_reward_to_the_fourth(abs(rotation), 0, 90) * 2)
            return reward, done
        elif is_crashed_anywhere:
            done = True
            reward = -5 + norm_reward(dist_landing_spot, 0, 3000 ** 2) * 5
            return reward, done
        if done:
            reward += self.compute_reward(state)
        if abs(vs) > 80:
            reward = -0.5
        return reward, done


def norm_reward(feature, interval_low, interval_high) -> float:
    feature = np.clip(feature, interval_low, interval_high)
    return 1.0 - ((feature - interval_low) / (interval_high - interval_low))


def norm_reward_to_the_fourth(feature, interval_low, interval_high) -> float:
    feature = np.clip(feature, interval_low, interval_high)
    return 1.0 - (((feature - interval_low) / (interval_high - interval_low)) ** (1 / 4))


def action_2_min_max(old_rota: int) -> list:
    return [max(old_rota - 15, -90), min(old_rota + 15, 90)]


def action_1_min_max(old_power_thrust: int) -> list:
    return [max(old_power_thrust - 1, 0), min(old_power_thrust + 1, 4)]


def actions_min_max(action: list) -> tuple[list, list]:
    return action_2_min_max(action[0]), action_1_min_max(action[1])


def limit_actions(old_rota: int, old_power_thrust: int, action: list) -> list:
    range_rotation = action_2_min_max(old_rota)
    rot = action[0] if range_rotation[0] <= action[0] <= range_rotation[1] else min(
        max(action[0], range_rotation[0]), range_rotation[1])
    range_thrust = action_1_min_max(old_power_thrust)
    thrust = action[1] if range_thrust[0] <= action[1] <= range_thrust[1] else min(
        max(action[1], range_thrust[0]), range_thrust[1])
    return [rot, thrust]