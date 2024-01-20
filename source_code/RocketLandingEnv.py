import math
import numpy as np

# from source_code.graph_handler import create_graph, display_graph
from source_code.math_utils import distance_to_line, distance_2, calculate_intersection, do_segments_intersect, \
    do_segments_intersect_vector


class RocketLandingEnv:
    def __init__(self):
        self.n_intermediate_path = 6
        initial_pos = [6500, 2000]
        initial_hs = 0
        initial_vs = 0
        initial_rotation = 0
        initial_thrust = 0
        self.initial_fuel = 1200
        self.rewards_episode = []
        self.prev_shaping = None
        self.reward_plot = []
        self.trajectory_plot = []
        self.surface = self.parse_planet_surface()
        self.surface_segments = list(zip(self.surface[:-1], self.surface[1:]))
        if initial_pos[0] > 7000 / 2:
            self.surface_segments.reverse()
        self.landing_spot = self.find_landing_spot(self.surface)
        self.middle_landing_spot = np.mean(self.landing_spot, axis=0)
        self.path_to_the_landing_spot = self.search_path(initial_pos)
        self.checkpoint = 0

        self.path_to_the_landing_spot = np.array(
            [np.array([x, y + 1000]) if i < len(self.path_to_the_landing_spot) - 1 else np.array([x, y]) for i, (x, y) in
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
        self.ax_trajectories = ax_trajectories
        self.ax_terminal_state_rewards = ax_terminal_state_rewards
        create_graph(self.surface, 'Landing on Mars', ax_trajectories, self.path_to_the_landing_spot)

    @staticmethod
    def parse_planet_surface():
        input_file = '''
18
0 1800
300 1200
1000 1550
2000 1200
2500 1650
3700 220
4700 220
4750 1000
4700 1650
4000 1700
3700 1600
3750 1900
4000 2100
4900 2050
5100 1000
5500 500
6200 800
6999 600
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

    def step(self, action_to_do: np.ndarray) -> tuple[np.ndarray, float, bool, bool, bool]:
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

        if distance_2(new_pos, self.path_to_the_landing_spot[self.checkpoint]) < 500 * 500:
            if self.checkpoint < len(self.path_to_the_landing_spot) - 1:
                self.checkpoint += 1
                print("Checkpoint Next", self.path_to_the_landing_spot[self.checkpoint])

        dist_to_path = self.get_distance_to_path(new_pos, self.path_to_the_landing_spot)

        new_state = [new_pos[0], new_pos[1], new_horizontal_speed, new_vertical_speed,
                     remaining_fuel, rotation, thrust, dist_to_landing_spot,
                     dist_to_surface, dist_to_path]
        return np.array(new_state)

    def render(self):
        self.reward_plot.append(np.sum(self.rewards_episode))
        display_graph(self.trajectory_plot, 0, self.ax_trajectories)
        self.trajectory_plot = []
        self.rewards_episode = []

    @staticmethod
    def _compute_reward(state: np.ndarray) -> tuple[float, bool, bool]:
        terminated, truncated = False, False
        x, y, hs, vs, remaining_fuel, rotation, thrust, dist_landing_spot, dist_surface, dist_path = state

        is_successful_landing = dist_landing_spot < 1 and rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20
        is_crashed_on_landing_spot = dist_landing_spot < 2
        is_crashed_anywhere = y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or dist_surface < 2 or remaining_fuel < 4
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
            truncated = True
            reward -= 100
        return reward, terminated, truncated

    def generate_random_action(self, old_rota: int, old_power_thrust: int) -> np.ndarray:
        rotation_limits = self._generate_action_limits(old_rota, 15, -90, 90)
        thrust_limits = self._generate_action_limits(old_power_thrust, 1, 0, 4)
        random_rotation = np.random.randint(rotation_limits[0], rotation_limits[1] + 1)
        random_thrust = np.random.randint(thrust_limits[0], thrust_limits[1] + 1)
        return np.array([random_rotation, random_thrust], dtype=int)

    def search_path(self, initial_pos):
        path = []
        intersect = False
        intersect_index = 0
        for idx, segment in enumerate(self.surface_segments):
            if do_segments_intersect([initial_pos, self.middle_landing_spot], segment):
                intersect = True
                intersect_index = idx
                break
        if not intersect:
            path.append(initial_pos)
            path.append(self.middle_landing_spot)
            return path
        idx = intersect_index
        while idx < len(self.surface_segments):
            segment = self.surface_segments[idx]
            high_point = segment[0]
            if do_segments_intersect_vector([high_point, self.middle_landing_spot], self.surface_segments):
                idx += 1
            else:
                path.extend(np.round(np.linspace(initial_pos, high_point, 5)).astype(int))
                path.extend(np.round(np.linspace(high_point, self.middle_landing_spot, 5)).astype(int))
                for item in path:
                    print(item)
                # print(path)
                # exit(0)
                return path

    def get_distance_to_path(self, new_pos, path_to_the_landing_spot):
        if self.checkpoint >= len(self.path_to_the_landing_spot):
            self.checkpoint = len(self.path_to_the_landing_spot) - 1
        highest = self.path_to_the_landing_spot[self.checkpoint]

        if highest is None:
            highest = path_to_the_landing_spot[-1]
        # print(highest, new_pos)
        if highest[0] != 6500 and highest[1] != 3000:
            print(highest, new_pos)
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

import matplotlib
from matplotlib import pyplot as plt, cm
matplotlib.use('Qt5Agg')


def create_graph(line, title: str, ax, path_to_the_landing_spot):
    ax.set_xlim(0, 7000)
    ax.set_ylim(0, 3000)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(title)
    x, y = line[:, 0], line[:, 1]
    ax.plot(x, y, marker='o', label='Mars Surface')
    # Plot the path to the landing spot
    path_x, path_y = path_to_the_landing_spot[:, 0], path_to_the_landing_spot[:, 1]
    ax.plot(path_x, path_y, marker='x', linestyle='--', label='Path to Landing Spot')

    ax.legend()
    ax.grid(True)


def display_graph(trajectories, id_lines_color: int, ax):
    cmap = cm.get_cmap('Set1')
    color = cmap(id_lines_color % 9)

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() == 'Rocket':
            line.remove()

    x_values = [trajectory[0] for trajectory in trajectories]
    y_values = [trajectory[1] for trajectory in trajectories]

    ax.plot(x_values, y_values, marker='o', markersize=2, color=color, label='Rocket')
    plt.pause(0.001)