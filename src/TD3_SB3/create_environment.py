import math
import random
from itertools import product
from typing import List, Union

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Box
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from shapely import LineString, Point, MultiPoint

from src.TD3_SB3.graph_handler import create_graph, display_graph, plot_terminal_state_rewards
from src.PPO_to_remove.math_utils import distance_squared_to_line


class RocketLandingEnv(gymnasium.Env):
    def __init__(self):
        self.reward_plot = []
        self._i_step = 0
        self.trajectory_plot = []
        surface_points = self.parse_planet_surface()
        self.surface = LineString(surface_points.geoms)
        self.landing_spot = self.find_landing_spot(surface_points)
        initial_pos = [2500, 2500]
        self.initial_fuel = 550
        self.initial_state = np.array([
            initial_pos[0],  # x
            initial_pos[1],  # y
            0,  # horizontal speed
            0,  # vertical speed
            self.initial_fuel,  # fuel remaining
            0,  # rotation
            0,  # thrust power
            distance_squared_to_line(initial_pos, self.landing_spot),  # distance to landing spot
            distance_squared_to_line(initial_pos, self.surface)  # distance to surface
        ])
        self.state_intervals = [
            [0, 7000],  # x
            [0, 3000],  # y
            [-150, 150],  # vs
            [-150, 150],  # hs
            [0, self.initial_fuel],  # fuel remaining
            [-90, 90],  # rot
            [0, 4],  # thrust
            [0, 3000 ** 2],  # distance squared landing_spot
            [0, 3000 ** 2]  # distance squared surface
        ]
        self.state = self.initial_state
        self.action_constraints = [15, 1]
        self.gravity = 3.711

        self.observation_space = spaces.Box(low=np.array([0.0] * 9), high=np.array([1.0] * 9))

        action_1_bounds = [-1, 1]
        action_2_bounds = [-1, 1]
        action_space_dimension = 2

        self.action_space = spaces.Box(
            low=np.array([action_1_bounds[0], action_2_bounds[0]]),
            high=np.array([action_1_bounds[1], action_2_bounds[1]]),
            shape=(action_space_dimension,)
        )
        fig, (ax_terminal_state_rewards, ax_trajectories) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig = fig
        # self.ax_rewards = ax_mean_rewards
        self.ax_trajectories = ax_trajectories
        self.ax_terminal_state_rewards = ax_terminal_state_rewards
        create_graph(self.surface, 'Landing on Mars', ax_trajectories)

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
        return np.array([(val - interval[0]) / (interval[1] - interval[0])
                         for val, interval in zip(raw_state, self.state_intervals)])

    def denormalize_state(self, normalized_state: list):
        return np.array([val * (interval[1] - interval[0]) + interval[0]
                         for val, interval in zip(normalized_state, self.state_intervals)])

    def _get_obs(self):
        return self.state

    def reset(self, seed=None, options=None):
        self.state = self.normalize_state(self.initial_state)
        return self._get_obs(), {}

    def step(self, action_to_do_input):
        action_to_do = np.copy(action_to_do_input)
        action_to_do[0] = action_to_do[0] * 90
        # action_to_do[1] = action_to_do[1] * 4
        action_to_do[1] = (action_to_do[1] + 1) / 2 * 4

        action_to_do = np.round(action_to_do)

        # print("Step done with action :", action_to_do_input, action_to_do)

        tmp = self.denormalize_state(self.state)
        # self.state = self.compute_next_state(self.state, action_to_do)
        self.state = self.compute_next_state(tmp, action_to_do)
        self.state = self.normalize_state(self.state)
        obs = self._get_obs()
        # reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        reward, terminated = self.reward_function(obs)
        self.trajectory_plot.append(self.denormalize_state(obs))
        if terminated:
            self.reward_plot.append(reward)
            plot_terminal_state_rewards(self.reward_plot, self.ax_terminal_state_rewards)
            display_graph(self.trajectory_plot, 0, self.ax_trajectories)
            self.trajectory_plot = []
        # if self._i_step % 1000 == 0:
        #     self.save_model()
        # self._i_step += 1

        return self._get_obs(), reward, terminated, False, {}

    def is_done(self, state):
        state = self.denormalize_state(state)
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface = state
        is_successful_landing = (dist_landing_spot < 1 and rotation == 0 and
                                 abs(vs) <= 40 and abs(hs) <= 20)
        is_crashed_anywhere = (y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or
                               dist_surface < 1 or remaining_fuel < -4)
        is_crashed_on_landing_spot = dist_landing_spot < 1
        if is_successful_landing or is_crashed_anywhere or is_crashed_on_landing_spot:
            return True
        else:
            return False

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
        return np.array(new_state)

    def render(self):
        pass

    def close(self):
        pass

    def convert_to_bit_vector(self, state: Union[int, np.ndarray], batch_size: int) -> np.ndarray:
        """
        Convert to bit vector if needed.

        :param state: The state to be converted, which can be either an integer or a numpy array.
        :param batch_size: The batch size.
        :return: The state converted into a bit vector.
        """
        # Convert back to bit vector
        if isinstance(state, int):
            bit_vector = np.array(state).reshape(batch_size, -1)
            # Convert to binary representation
            bit_vector = ((bit_vector[:, :] & (1 << np.arange(len(self.state)))) > 0).astype(int)
        else:
            bit_vector = np.array(state).reshape(batch_size, -1)
        return bit_vector

    def compute_reward(self, achieved_goal, desired_goal, info):
        if isinstance(achieved_goal, int):
            batch_size = 1
        else:
            batch_size = achieved_goal.shape[0] if len(achieved_goal.shape) > 1 else 1

        desired_goal = self.convert_to_bit_vector(desired_goal, batch_size)
        achieved_goal = self.convert_to_bit_vector(achieved_goal, batch_size)
        distance = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # t = -(distance > 0.1).astype(np.float32)
        # print(t)
        return -(distance > 0).astype(np.float32)

    def reward_function(self, state: list) -> tuple[float, bool]:
        state = self.denormalize_state(state)
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface = state
        is_successful_landing = (dist_landing_spot < 1 and rotation == 0 and
                                 abs(vs) <= 40 and abs(hs) <= 20)
        is_crashed_anywhere = (y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or
                               dist_surface < 1 or remaining_fuel < -4)
        is_crashed_on_landing_spot = dist_landing_spot < 1

        reward = 0
        done = False
        if is_successful_landing:
            # print("SUCCESSFUL LANDING !", state)
            print('SUCCESS')
            exit(80)
            done = True
            reward = 3 + norm_reward(self.initial_fuel - remaining_fuel, 0, self.initial_fuel) * 5
            return reward, done
        elif is_crashed_on_landing_spot:
            formatted_list = [f"{label}: {round(value):04d}" for label, value in
                              zip(['vs', 'hs', 'rotation'], [vs, hs, rotation])]
            done = True
            reward = (norm_reward(abs(vs), 40, 150) +
                      norm_reward(abs(hs), 20, 150) +
                      # (1 if abs(rotation) == 0.0 else 0)
                      norm_reward_to_the_fourth(abs(rotation), 0, 90)
                      )
            print('Crash on landing side', formatted_list, reward, norm_reward(abs(vs), 40, 150), norm_reward(abs(hs), 20, 150), norm_reward_to_the_fourth(abs(rotation), 0, 90))
            return reward, done
            # return reward, done
        elif is_crashed_anywhere:
            print("crash anywhere", [x, y])
            done = True
            reward = -1 + norm_reward(dist_landing_spot, 0, 3000 ** 2)
            # return reward, done
            return -1, done
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
