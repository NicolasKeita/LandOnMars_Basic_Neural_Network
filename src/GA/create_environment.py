import math
import random
import numpy as np
from matplotlib import pyplot as plt

from src.GA.graph_handler import create_graph, display_graph, plot_terminal_state_rewards
from src.GA.math_utils import distance_to_line, distance_2, calculate_intersection


class RocketLandingEnv:
    def __init__(self):
        self.i_intermediate_path = None
        initial_pos = [6500, 2800]
        initial_vs = 0
        initial_hs = -90
        initial_rotation = 90
        initial_thrust = 0
        self.initial_fuel = 750
        self.rewards_episode = []
        self.prev_shaping = None
        self.reward_plot = []
        self.trajectory_plot = []
        self.surface = self.parse_planet_surface()
        self.surface_segments = [(self.surface[i], self.surface[i + 1]) for i in range(len(self.surface) - 1)]
        self.landing_spot = self.find_landing_spot(self.surface)
        self.middle_landing_spot = np.mean(self.landing_spot, axis=0)
        self.path_to_the_landing_spot = self.search_path(initial_pos, self.surface, self.landing_spot, [])

        self.path_to_the_landing_spot = np.array(
            [np.array([x, y + 300]) if i < len(self.path_to_the_landing_spot) - 1 else np.array([x, y]) for i, (x, y) in
             enumerate(self.path_to_the_landing_spot)])
        surface_segments = [self.surface[i:i + 2] for i in range(len(self.surface) - 1)]

        self.initial_state = np.array([
            initial_pos[0],  # x
            initial_pos[1],  # y
            initial_hs,  # horizontal speed
            initial_vs,  # vertical speed
            self.initial_fuel,  # fuel remaining
            initial_rotation,  # rotation
            initial_thrust,  # thrust power
            distance_to_line(initial_pos, [self.landing_spot]),  # distance to landing spot
            distance_to_line(initial_pos, surface_segments),  # distance to surface
            0
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
        7
        0 100
        1000 500
        1500 1500
        3000 1000
        4000 150
        5500 150
        6999 800
        '''
        return np.fromstring(input_file, sep='\n', dtype=int)[1:].reshape(-1, 2)
        # return MultiPoint(points_coordinates), points_coordinates

    @staticmethod
    def find_landing_spot(planet_surface: np.ndarray) -> np.ndarray:
        points = planet_surface
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            if y1 == y2:
                return np.array([[x1, y1], [x2, y2]])
        raise Exception('No landing site on test-case data')

    def reset(self, seed=None, options=None):
        self.prev_shaping = None
        self.trajectory_plot = []
        self.state = self.initial_state
        return self.state, {}

    def step(self, action_to_do: np.ndarray):
        self.state = self._compute_next_state(self.state, action_to_do)
        reward, terminated, truncated = self._compute_reward(self.state)
        self.trajectory_plot.append(self.state)
        self.rewards_episode.append(reward)
        # self.render()
        # if terminated or truncated:
        #     self.render()
        return self.state, reward, terminated, truncated, {'side': self.straighten_info(self.state)}

    def _compute_next_state(self, state, action: np.ndarray):
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
        new_pos = calculate_intersection(x, y, new_pos, self.surface)
        remaining_fuel = max(remaining_fuel - thrust, 0)
        surface_segments = [self.surface[i:i + 2] for i in range(len(self.surface) - 1)]
        dist_to_landing_spot = distance_to_line(new_pos, [self.landing_spot])
        dist_to_surface = distance_to_line(new_pos, surface_segments)
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

    def _compute_reward(self, state: np.ndarray) -> tuple[float, bool, bool]:
        terminated, truncated = False, False
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface, dist_path = state
        if dist_landing_spot < 2000 ** 2:
            shaping = (
                    -100 * (1 - norm_reward(dist_landing_spot, 0, 7000 ** 2))
                    - 50 * (1 - norm_reward(abs(vs), 39, 150))
                    - 50 * (1 - norm_reward(abs(hs), 19, 150))
            )
        else:
            shaping = -100 * (1 - norm_reward(dist_path, 0, 1000 ** 2))
        reward = shaping - self.prev_shaping if self.prev_shaping is not None else 0
        self.prev_shaping = shaping
        # print("Reward before thrust:", reward)
        # reward -= thrust * 0.30
        # print("reward = ", reward)
        is_successful_landing = dist_landing_spot < 1 and rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20
        is_crashed_on_landing_spot = dist_landing_spot < 1
        is_crashed_anywhere = y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or dist_surface < 1 or remaining_fuel < 5

        reward = norm_reward(dist_landing_spot, 0, 7000 ** 2) + norm_reward(abs(vs), 39, 150) + norm_reward(abs(hs), 19, 150)

        if is_successful_landing:
            print("SUCCESSFUL LANDING !", state)
            terminated, reward = True, +10000 + remaining_fuel * 100
        elif is_crashed_on_landing_spot:
            formatted_list = [f"{label}: {value:04f}" for label, value in
                              zip(['vs', 'hs', 'rotation'], [vs, hs, rotation])]
            print('Crash on landing side', formatted_list)
            # terminated = True
            terminated = True
            # reward = -100
            # reward = 1000 + norm_reward(abs(vs), 39, 150) * 100 + norm_reward(abs(hs), 19, 150) * 100
        elif is_crashed_anywhere:
            print("Crash anywhere", x, y)
            truncated = True
            # truncated, reward = True, -100
        return reward, terminated, truncated

    def generate_random_action(self, old_rota: int, old_power_thrust: int) -> np.ndarray:
        rotation_limits = self._generate_action_limits(old_rota, 15, -90, 90)
        thrust_limits = self._generate_action_limits(old_power_thrust, 1, 0, 4)
        random_rotation = np.random.randint(rotation_limits[0], rotation_limits[1] + 1)
        random_thrust = np.random.randint(thrust_limits[0], thrust_limits[1] + 1)
        return np.array([random_rotation, random_thrust], dtype=int)

    def search_path(self, initial_pos, surface, landing_spot, my_path):
        def do_segments_intersect(segment1, segment2):
            x1, y1 = segment1[0]
            x2, y2 = segment1[1]
            x3, y3 = segment2[0]
            x4, y4 = segment2[1]

            # Check if segments have the same origin
            if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4) or (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
                return False

            # Check if the segments intersect using a basic algorithm
            def orientation(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0
                return 1 if val > 0 else 2

            def on_segment(p, q, r):
                return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

            o1 = orientation((x1, y1), (x2, y2), (x3, y3))
            o2 = orientation((x1, y1), (x2, y2), (x4, y4))
            o3 = orientation((x3, y3), (x4, y4), (x1, y1))
            o4 = orientation((x3, y3), (x4, y4), (x2, y2))

            if (o1 != o2 and o3 != o4) or (o1 == 0 and on_segment((x1, y1), (x3, y3), (x2, y2))) or (
                    o2 == 0 and on_segment((x1, y1), (x4, y4), (x2, y2))) or (
                    o3 == 0 and on_segment((x3, y3), (x1, y1), (x4, y4))) or (
                    o4 == 0 and on_segment((x3, y3), (x2, y2), (x4, y4))):
                if not on_segment((x1, y1), (x2, y2), (x3, y3)) and not on_segment((x1, y1), (x2, y2), (x4, y4)):
                    return True

            return False

        path = []
        # check if there are any obstacles in the path
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
            t = np.linspace(initial_pos, self.middle_landing_spot, 5)
            if len(my_path) == 0:
                return t
            else:
                my_path = my_path[:-1, :]
                return np.concatenate((my_path, t))
        else:
            path[0][1] = path[0][1]
            t = np.linspace(initial_pos, path[0], 5)
            return self.search_path(path[0], surface, landing_spot, t)

    def get_distance_to_path(self, new_pos, path_to_the_landing_spot):
        highest = None

        for i, point in enumerate(path_to_the_landing_spot):
            # print(np.array_equal(point, path_to_the_landing_spot[0]), point, path_to_the_landing_spot[0])
            if new_pos[1] >= point[1]:
                highest = point
                self.i_intermediate_path = i
                break
        if highest is None:
            highest = path_to_the_landing_spot[-1]
            self.i_intermediate_path = len(path_to_the_landing_spot) - 1
        # self.i_intermediate_path = np.where(np.array_equal(path_to_the_landing_spot, highest))[0]
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
