import math
import numpy as np
from shapely import LineString, Point, MultiPoint

from src.hyperparameters import limit_actions, GRAVITY, actions_min_max
from src.math_utils import distance_squared_to_line


class RocketLandingEnv:
    def __init__(self):
        surface_points = self.parse_planet_surface()
        self.surface = LineString(surface_points.geoms)
        self.landing_spot = self.find_landing_spot(surface_points)
        self.initial_state = [
            2500,  # x
            2500,  # y
            0,  # horizontal speed
            0,  # vertical speed
            20000,  # fuel remaining
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
            [-10, 20000],  # fuel remaining
            [-90, 90],  # rot
            [0, 4],  # thrust
            [0, 3000 ** 2],  # distance squared landing_spot
            [0, 3000 ** 2]  # distance squared surface
        ]
        self.state = self.initial_state
        self.action_constraints = [15, 1]
        self.action_space_dimension = 2

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
        # print(action_2, sigmoid(normalized_actions[1]), normalized_actions[1])
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

    def get_action_constraints(self, normalized_previous_action: list):
        if normalized_previous_action is None:
            return [self.normalize_action([-15, 0]), self.normalize_action([15, 1])]
        action = self.denormalize_action(normalized_previous_action)
        legal_min_max = actions_min_max(action)
        minimum = self.normalize_action((legal_min_max[0][0], legal_min_max[1][0]))
        maximum = self.normalize_action((legal_min_max[0][1], legal_min_max[1][1]))
        return [minimum, maximum]

    def reset(self):
        self.state = self.initial_state
        return self.normalize_state(self.state)

    def step(self, action: tuple[int, int]):
        self.state = self.compute_next_state(self.state, action)
        reward, done = reward_function(self.state)
        next_state_normalized = self.normalize_state(self.state)
        return next_state_normalized, reward, done, None

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

        line1 = LineString(self.surface)
        line2 = LineString([curr_pos, new_pos])
        intersection: Point = line1.intersection(line2)
        if not intersection.is_empty and isinstance(intersection, Point):
            new_pos = np.array(intersection.xy).flatten()
        remaining_fuel = state[4] - thrust

        new_state = [new_pos[0],
                     new_pos[1],
                     new_horizontal_speed,
                     new_vertical_speed, remaining_fuel, rot,
                     thrust,
                     distance_squared_to_line(np.array(new_pos), self.landing_spot),
                     distance_squared_to_line(new_pos, self.surface)]
        return new_state


def norm_reward(feature, interval_low, interval_high) -> float:
    return max(0.0, 1.0 - abs(feature) / interval_high)


def compute_reward(state) -> float:
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
    # if (thrust == 4):
    #     print("thrust !")
    #     exit(11)
    if is_close_to_land:
        rotation_normalized = norm_reward(rotation, 0, 90) * rotation_scaling * dist_normalized
    else:
        rotation_normalized = 1
    # return vs_normalized
    return dist_normalized + hs_normalized + vs_normalized + rotation_normalized


def reward_function(state) -> (float, bool):
    x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface = state

    is_successful_landing = (dist_landing_spot < 1 and rotation == 0 and
                             abs(vs) <= 40 and abs(hs) <= 20)
    is_crashed = (y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or
                  dist_surface < 1 or remaining_fuel < -4)

    reward = compute_reward(state)
    done = False
    if is_successful_landing:
        print("SUCCESSFUL LANDING !")
        done = True
        reward += 10
        exit(0)
    elif is_crashed:
        print("Crash, ", state)
        done = True
        reward -= 10
    return reward, done
