import math
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from src.GA.graph_handler import create_graph, display_graph, plot_terminal_state_rewards
from src.GA.math_utils import distance_to_line, distance_2, calculate_intersection, do_segments_intersect


class RocketLandingEnv:
    def __init__(self):
        self.n_intermediate_path = 6
        initial_pos = [500, 2700]
        initial_hs = 100
        initial_vs = 0
        initial_rotation = -90
        initial_thrust = 0
        self.initial_fuel = 800
        self.rewards_episode = []
        self.prev_shaping = None
        self.reward_plot = []
        self.trajectory_plot = []
        self.surface = self.parse_planet_surface()
        self.surface_segments = list(zip(self.surface[:-1], self.surface[1:]))
        self.landing_spot = self.find_landing_spot(self.surface)
        self.middle_landing_spot = np.mean(self.landing_spot, axis=0)
        self.path_to_the_landing_spot = self.search_path(initial_pos, self.landing_spot, [])

        self.path_to_the_landing_spot = np.array(
            [np.array([x, y + 200]) if i < len(self.path_to_the_landing_spot) - 1 else np.array([x, y]) for i, (x, y) in
             enumerate(self.path_to_the_landing_spot)])


        self.initial_state = np.array([
            float(initial_pos[0]),  # x
            float(initial_pos[1]),  # y
            float(initial_hs),  # horizontal speed
            float(initial_vs),  # vertical speed
            float(self.initial_fuel),  # fuel remaining
            float(initial_rotation),  # rotation
            float(initial_thrust),  # thrust power
            float(distance_to_line(initial_pos[0], initial_pos[1], np.array([self.landing_spot]))),  # distance to landing spot
            float(distance_to_line(initial_pos[0], initial_pos[1], np.array(self.surface_segments))),  # distance to surface
            float(0)
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
            [0, 3000 ** 2],  # distance squared surface
            [0, 3000 ** 2]
        ]
        self.state = self.initial_state
        self.action_constraints = [15, 1]
        self.gravity = 3.711

        fig, (ax_terminal_state_rewards, ax_trajectories) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig = fig
        # plt.close(fig)
        # self.ax_rewards = ax_mean_rewards
        self.ax_trajectories = ax_trajectories
        self.ax_terminal_state_rewards = ax_terminal_state_rewards
        create_graph(self.surface, 'Landing on Mars', ax_trajectories, self.path_to_the_landing_spot)

    @staticmethod
    def parse_planet_surface():
        input_file = '''
20
0 1000
300 1500
350 1400
500 2000
800 1800
1000 2500
1200 2100
1500 2400
2000 1000
2200 500
2500 100
2900 800
3000 500
3200 1000
3500 2000
3800 800
4000 200
5000 200
5500 1500
6999 2800
        '''
        return np.fromstring(input_file, sep='\n', dtype=int)[1:].reshape(-1, 2)


    @staticmethod
    def find_landing_spot(planet_surface: np.ndarray) -> np.ndarray:
        for (x1, y1), (x2, y2) in zip(planet_surface, planet_surface[1:]):
            if y1 == y2:
                return np.array([[x1, y1], [x2, y2]])
        raise Exception('No landing site on test-case data')

    def reset(self, seed=None, options=None):
        self.prev_shaping = None
        self.state = self.initial_state
        return self.state, {}

    def step(self, action_to_do: np.ndarray[int, 1]) -> tuple[np.ndarray, float, bool, bool, bool]:
        self.state = self._compute_next_state(self.state, action_to_do)
        reward, terminated, truncated = self._compute_reward(self.state)
        self.trajectory_plot.append(self.state)
        self.rewards_episode.append(reward)
        return self.state, reward, terminated, truncated, False

    def _compute_next_state(self, state: np.ndarray, action: np.ndarray[int, 1]) -> np.ndarray:
        x, y, hs, vs, remaining_fuel, rotation, thrust, _, _, _ = state
        rotation, thrust = self.limit_actions(rotation, thrust, action)
        radians = math.radians(rotation)
        x_acceleration = math.sin(radians) * thrust
        y_acceleration = math.cos(radians) * thrust - self.gravity
        new_horizontal_speed = hs - x_acceleration
        new_vertical_speed = vs + y_acceleration
        new_x = x + hs - 0.5 * x_acceleration
        new_y = y + vs + 0.5 * y_acceleration
        new_pos = np.clip([new_x, new_y], [0, 0], [7000, 3000])
        new_pos = calculate_intersection(np.array([x, y]), new_pos, self.surface)
        remaining_fuel = max(remaining_fuel - thrust, 0)
        surface_segments = np.array([self.surface[i:i + 2] for i in range(len(self.surface) - 1)])
        dist_to_landing_spot = distance_to_line(new_pos[0], new_pos[1], np.array([self.landing_spot]))
        dist_to_surface = distance_to_line(new_pos[0], new_pos[1], surface_segments)
        dist_to_path = self.get_distance_to_path(new_pos, self.path_to_the_landing_spot)

        new_state = [new_pos[0], new_pos[1], new_horizontal_speed, new_vertical_speed,
                     remaining_fuel, rotation, thrust, dist_to_landing_spot,
                     dist_to_surface, dist_to_path]
        return np.array(new_state)

    def render(self):
        # return None
        self.reward_plot.append(np.sum(self.rewards_episode))
        plot_terminal_state_rewards(self.reward_plot, self.ax_terminal_state_rewards)
        display_graph(self.trajectory_plot, 0, self.ax_trajectories)
        self.trajectory_plot = []
        self.rewards_episode = []

    @staticmethod
    def _compute_reward(state: np.ndarray) -> tuple[float, bool, bool]:
        terminated, truncated = False, False
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface, dist_path = state

        is_successful_landing = dist_landing_spot < 1 and rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20
        is_crashed_on_landing_spot = dist_landing_spot < 1
        is_crashed_anywhere = y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or dist_surface < 1 or remaining_fuel < 4
        is_close_to_land = dist_landing_spot < 1500 ** 2

        if is_close_to_land:
            reward = (norm_reward(dist_landing_spot, 0, 7500 ** 2)
                      + 0.65 * norm_reward(abs(vs), 39, 140)
                      + 0.35 * norm_reward(abs(hs), 19, 140)
            )
        else:
            reward = (norm_reward(dist_path, 0, 7500 ** 2)
                      + 0.65 * norm_reward(abs(vs), 39, 150)
                      + 0.35 * norm_reward(abs(hs), 19, 150)
            )

        if is_successful_landing:
            print("SUCCESSFUL LANDING !", state)
            terminated, reward = True, +10000 + remaining_fuel * 100
        elif is_crashed_on_landing_spot:
            formatted_list = [f"{label}: {value:04f}" for label, value in
                              zip(['vs', 'hs', 'rotation'], [vs, hs, rotation])]
            print('Crash on landing side', formatted_list)
            terminated = True
            reward -= 100

        elif is_crashed_anywhere:
            print("Crash anywhere", x, y)
            truncated = True
            reward -= 100
        return reward, terminated, truncated

    def generate_random_action(self, old_rota: int, old_power_thrust: int) -> np.ndarray:
        rotation_limits = self._generate_action_limits(old_rota, 15, -90, 90)
        thrust_limits = self._generate_action_limits(old_power_thrust, 1, 0, 4)
        random_rotation = np.random.randint(rotation_limits[0], rotation_limits[1] + 1)
        random_thrust = np.random.randint(thrust_limits[0], thrust_limits[1] + 1)
        return np.array([random_rotation, random_thrust], dtype=int)

    def search_path(self, initial_pos, landing_spot, my_path):
        path = []
        for segment in self.surface_segments:
            segment1 = [initial_pos, self.middle_landing_spot]
            segment2 = segment
            if do_segments_intersect(segment1, segment2):
                if segment2[0][1] > segment2[1][1]:
                    path.append(segment2[0])
                elif segment2[0][1] < segment2[1][1]:
                    path.append(segment2[1])
                else:
                    path.append(random.choice([segment2[0], segment2[1]]))
                break
        if len(path) == 0:
            t = np.round(np.linspace(initial_pos, self.middle_landing_spot, self.n_intermediate_path)).astype(int)
            if len(my_path) == 0:
                return t
            else:
                my_path = my_path[:-1, :]
                return np.concatenate((my_path, t))
        else:
            path[0][1] = path[0][1]
            t = np.round(np.linspace(initial_pos, path[0], self.n_intermediate_path)).astype(int)
            return self.search_path(path[0], landing_spot, t)

    # def search_path(self, initial_pos: np.ndarray[int, 1], landing_spot, my_path):
    #     path = next(((s2[0] if s2[0][1] > s2[1][1] else s2[1]) if do_segments_intersect(
    #         [initial_pos, self.middle_landing_spot], s2) else None) for s2 in self.surface_segments)
    #
    #     if path is None:
    #         t = np.round(np.linspace(initial_pos, self.middle_landing_spot, self.n_intermediate_path)).astype(int)
    #         return t if len(my_path) == 0 else np.concatenate((my_path[:-1, :], t))
    #
    #     path[1] = path[1]
    #     return self.search_path(path, landing_spot,
    #                             np.round(np.linspace(initial_pos, path, self.n_intermediate_path)).astype(int))

    def get_distance_to_path(self, new_pos, path_to_the_landing_spot):
        highest = None

        for i, point in enumerate(path_to_the_landing_spot):
            if new_pos[1] >= point[1] and not distance_2(new_pos, point) < 25 ** 2:
            # if new_pos[1] >= point[1]:
                highest = point
                self.i_intermediate_path = i
                break
        if highest is None:
            highest = path_to_the_landing_spot[-1]
        return distance_2(highest, new_pos)

    def straighten_info(self, state):
        x = state[0]
        y = state[1]
        highest = None
        for point in self.path_to_the_landing_spot:
            if y >= point[1]:
                highest = point
                break
        if highest is None:
            highest = self.path_to_the_landing_spot[-1]
        if x < highest[0]:
            return -1
        elif x > highest[0]:
            return 1
        else:
            return 0

    @staticmethod
    def _generate_action_limits(center: int, delta: int, min_value: int, max_value: int) -> tuple[int, int]:
        return max(center - delta, min_value), min(center + delta, max_value)

    def limit_actions(self, old_rota: int, old_power_thrust: int, action: np.ndarray) -> tuple[int, int]:
        rotation_limits = self._generate_action_limits(old_rota, 15, -90, 90)
        thrust_limits = self._generate_action_limits(old_power_thrust, 1, 0, 4)

        rotation = np.clip(action[0], *rotation_limits)
        thrust = np.clip(action[1], *thrust_limits)
        return rotation, thrust


def norm_reward(feature, interval_low, interval_high) -> float:
    feature = np.clip(feature, interval_low, interval_high)
    return 1.0 - ((feature - interval_low) / (interval_high - interval_low))
