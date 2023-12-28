import sys
import time
from typing import Tuple

import numpy as np
import math
import random

from numpy import ndarray


class GeneticAlgorithm:
    def __init__(self, env):
        self.env: RocketLandingEnv = env
        self.horizon = 15
        self.offspring_size = 9
        self.n_elites = 3
        self.n_heuristic_guides = 3
        self.mutation_rate = 0.4
        self.population_size = self.offspring_size + self.n_elites + self.n_heuristic_guides
        self.population = self.init_population(self.env.initial_state[5], self.env.initial_state[6])

        self.parents = None

    def crossover(self, population_survivors: np.ndarray, offspring_size: int) -> np.ndarray:
        offspring = []
        while len(offspring) < offspring_size:
            indices = np.random.randint(population_survivors.shape[0], size=(2))
            parents = population_survivors[indices]
            policy = np.zeros((self.horizon, 2), dtype=int)
            for i in range(self.horizon):
                offspring_rotation = my_random_int(parents[0, i, 0], parents[1, i, 0])
                offspring_thrust = my_random_int(parents[0, i, 1], parents[1, i, 1])
                offspring_rotation = np.clip(offspring_rotation, -90, 90)
                offspring_thrust = np.clip(offspring_thrust, 0, 4)
                policy[i] = [offspring_rotation, offspring_thrust]
            offspring.append(policy)
        return np.array(offspring)

    def mutation(self, population: np.ndarray) -> np.ndarray:
        individual = population[np.random.randint(population.shape[0])]
        for action in individual:
            action[1] = 4
        for individual in population:
            for action in individual:
                if np.random.rand() < self.mutation_rate:
                    action[0] += np.random.randint(-15, 16)
                    action[1] += np.random.randint(-1, 2)
                    action[0] = np.clip(action[0], -90, 90)
                    action[1] = np.clip(action[1], 0, 4)
        return population

    def heuristic(self, curr_initial_state: np.ndarray) -> np.ndarray:
        heuristics_guides = np.zeros((3, self.horizon, 2), dtype=int)
        x = curr_initial_state[0]
        y = curr_initial_state[1]
        angle = np.arctan2(self.env.middle_landing_spot[1] - y, self.env.middle_landing_spot[0] - x)
        angle_degrees = np.degrees(angle)
        for action in heuristics_guides[0]:
            action[1] = 4
            action[0] = np.clip(round(-angle_degrees), -90, 90)
        for action in heuristics_guides[1]:
            action[1] = 4
            action[0] = np.clip(round(angle_degrees), -90, 90)
        for action in heuristics_guides[2]:
            action[1] = 4
            action[0] = 0
        return heuristics_guides

    def learn(self, time_available):
        self.env.reset()
        curr_initial_state = self.env.initial_state
        self.population = self.init_population(curr_initial_state[5], curr_initial_state[6], self.parents)
        start_time = time.time()
        while True:
            rewards = np.array([self.rollout(individual) for individual in self.population])
            self.parents = self.selection(rewards, self.population, self.n_elites)
            if (time.time() - start_time) * 1000 >= time_available:
                break
            heuristic_guides = self.heuristic(curr_initial_state)
            heuristic_guides = np.array(
                [item for item in heuristic_guides if not np.any(np.all(item == self.parents, axis=(1, 2)))])
            offspring_size = self.offspring_size + self.n_heuristic_guides - len(heuristic_guides)
            offspring = self.crossover(self.parents, offspring_size)
            offspring = self.mutation(offspring)
            offspring = self.mutation_heuristic(offspring, curr_initial_state[7])
            self.population = np.concatenate((offspring, self.parents, heuristic_guides)) if len(
                heuristic_guides) > 0 else np.concatenate((offspring, self.parents))
        best_individual = self.parents[-1]
        self.env.reset()
        action_to_do = self.final_heuristic_verification(best_individual[0], curr_initial_state)
        next_state, _, _, _, _ = self.env.step(action_to_do)
        self.env.initial_state = next_state
        self.parents = self.parents[:, 1:, :]
        last_elements_tuple = self.parents[:, -1, :]
        self.parents = np.concatenate(
            (self.parents, np.array([self.env.generate_random_action(*item) for item in last_elements_tuple])[:, np.newaxis, :]),
            axis=1)
        print(f"{action_to_do[0]} {action_to_do[1]}")

    def rollout(self, policy: np.ndarray) -> float:
        self.env.reset()
        reward = 0
        for action in policy:
            _, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return reward

    def init_population(self, previous_rotation: int, previous_thrust: int, parents=None) -> np.ndarray:
        population = np.zeros((self.population_size, self.horizon, 2), dtype=int)
        num_parents = None
        if parents is not None:
            num_parents = len(parents)
            population[:num_parents] = parents
        for i in range(num_parents if parents is not None else 0, self.population_size):
            population[i] = self.generate_random_individual(previous_rotation, previous_thrust)
        return population

    def generate_random_individual(self, previous_rotation: int, previous_thrust: int) -> np.ndarray[int, 2]:
        individual = np.zeros((self.horizon, 2), dtype=int)
        random_action = np.array([previous_rotation, previous_thrust])
        for i in range(self.horizon):
            random_action = self.env.generate_random_action(random_action[0], random_action[1])
            individual[i] = random_action
        return individual

    def replace_duplicates_with_random(self, population, previous_rotation, previous_thrust):
        _, unique_indices = np.unique(population, axis=0, return_index=True)
        duplicate_indices = np.setdiff1d(np.arange(population.shape[0]), unique_indices)
        for i in duplicate_indices:
            population[i] = self.generate_random_individual(previous_rotation, previous_thrust)
        return population

    @staticmethod
    def mutation_heuristic(population, dist_landing_spot):
        if dist_landing_spot < 300 ** 2:
            individual = population[np.random.randint(population.shape[0])]
            for action in individual:
                action[0] = 0
        return population

    @staticmethod
    def final_heuristic_verification(action_to_do: np.ndarray, state: np.ndarray) -> np.ndarray:
        rotation = state[5]
        if abs(rotation - action_to_do[0]) > 110:
            action_to_do[1] = 0
        return action_to_do

    @staticmethod
    def selection(population, rewards, n_parents):
        sorted_indices = np.argsort(rewards)
        parents = population[sorted_indices[-n_parents:]]
        return parents


def my_random_int(a, b):
    if a == b:
        return a
    else:
        return np.random.randint(min(a, b), max(a, b))


def distance_2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def distance_to_line(x: float, y: float, line_segments: np.ndarray) -> float:
    min_distance_squared = float('inf')

    for segment in line_segments:
        (x1, y1), (x2, y2) = segment.tolist()
        dx, dy = x2 - x1, y2 - y1

        dot_product = (x - x1) * dx + (y - y1) * dy
        t = max(0, min(1, dot_product / (dx ** 2 + dy ** 2)))
        closest_point = (x1 + t * dx, y1 + t * dy)

        segment_distance_squared = distance_2((x, y), closest_point)
        min_distance_squared = min(min_distance_squared, segment_distance_squared)

    return min_distance_squared


def calculate_intersection(previous_pos: np.ndarray, new_pos: np.ndarray, surface: np.ndarray) -> np.ndarray:
    x1, y1 = previous_pos
    x2, y2 = new_pos

    x3, y3 = surface[:-1, 0], surface[:-1, 1]
    x4, y4 = surface[1:, 0], surface[1:, 1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    mask = denominator != 0

    t = np.empty_like(denominator)
    u = np.empty_like(denominator)

    t[mask] = ((x1 - x3[mask]) * (y3[mask] - y4[mask]) - (y1 - y3[mask]) * (x3[mask] - x4[mask])) / denominator[mask]
    u[mask] = -((x1 - x2) * (y1 - y3[mask]) - (y1 - y2) * (x1 - x3[mask])) / denominator[mask]
    intersected_mask = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)
    if np.any(intersected_mask):
        intersection_x = x1 + t[intersected_mask] * (x2 - x1)
        intersection_y = y1 + t[intersected_mask] * (y2 - y1)
        new_pos = np.array([intersection_x[0], intersection_y[0]])
    return new_pos


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    return (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1]))


def do_segments_intersect(segment1, segment2):
    x1, y1 = segment1[0]
    x2, y2 = segment1[1]
    x3, y3 = segment2[0]
    x4, y4 = segment2[1]
    if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4) or (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
        return False

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

class RocketLandingEnv:
    def __init__(self, initial_state: list[int], surface: np.ndarray):
        self.n_intermediate_path = 6
        initial_pos = [initial_state[0], initial_state[1]]
        initial_hs = initial_state[2]
        initial_vs = initial_state[3]
        initial_rotation = initial_state[5]
        initial_thrust = initial_state[6]
        self.initial_fuel = initial_state[4]
        self.rewards_episode = []
        self.prev_shaping = None
        self.surface = surface
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
        self.state = self.initial_state
        self.action_constraints = [15, 1]
        self.gravity = 3.711


    @staticmethod
    def find_landing_spot(planet_surface: np.ndarray) -> np.ndarray:
        for (x1, y1), (x2, y2) in zip(planet_surface, planet_surface[1:]):
            if y1 == y2:
                return np.array([[x1, y1], [x2, y2]])
        raise Exception('No landing site on test-case data')

    def reset(self) -> tuple[np.ndarray, bool]:
        self.state = self.initial_state
        return self.state, False

    def step(self, action_to_do: np.ndarray) -> tuple[ndarray, float, bool, bool, bool]:
        self.state = self._compute_next_state(self.state, action_to_do)
        reward, terminated, truncated = self._compute_reward(self.state)
        return self.state, reward, terminated, truncated, False

    def _compute_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
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
        dist_to_landing_spot = distance_to_line(new_pos[0], new_pos[1], [self.landing_spot])
        dist_to_surface = distance_to_line(new_pos[0], new_pos[1], self.surface_segments)
        dist_to_path = self.get_distance_to_path(new_pos, self.path_to_the_landing_spot)

        new_state = np.array([new_pos[0], new_pos[1], new_horizontal_speed, new_vertical_speed,
                     remaining_fuel, rotation, thrust, dist_to_landing_spot,
                     dist_to_surface, dist_to_path])
        return new_state

    @staticmethod
    def _compute_reward(state: np.ndarray) -> tuple[float, bool, bool]:
        x, y, hs, vs, remaining_fuel, rotation, _, dist_landing_spot, dist_surface, dist_path = state
        is_successful_landing = dist_landing_spot < 1 and rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20
        is_crashed_on_landing_spot = dist_landing_spot < 1
        is_crashed_anywhere = y <= 1 or y >= 3000 - 1 or x <= 1 or x >= 7000 - 1 or dist_surface < 1 or remaining_fuel < 4
        is_close_to_land = dist_landing_spot < 1500 ** 2
        dist_path = dist_landing_spot if is_close_to_land else dist_path
        reward = (norm_reward(dist_path, 0, 7500 ** 2)
                    + 0.65 * norm_reward(abs(vs), 39, 140)
                    + 0.35 * norm_reward(abs(hs), 19, 140)
        )
        if is_successful_landing:
            return 1000 + remaining_fuel * 100, True, False
        elif is_crashed_on_landing_spot:
            return reward - 100, True, False
        elif is_crashed_anywhere:
            return reward - 100, False, True
        else:
            return reward, False, False

    def generate_random_action(self, old_rota: int, old_power_thrust: int) -> np.ndarray:
        rotation_limits = self._generate_action_limits(old_rota, 15, -90, 90)
        thrust_limits = self._generate_action_limits(old_power_thrust, 1, 0, 4)
        random_rotation = np.random.randint(rotation_limits[0], rotation_limits[1] + 1)
        random_thrust = np.random.randint(thrust_limits[0], thrust_limits[1] + 1)
        return np.array([random_rotation, random_thrust], dtype=int)

    def search_path(self, initial_pos, landing_spot, my_path):
        path = next(((s2[0] if s2[0][1] > s2[1][1] else s2[1]) if do_segments_intersect(
            [initial_pos, self.middle_landing_spot], s2) else None) for s2 in self.surface_segments)

        if path is None:
            t = np.round(np.linspace(initial_pos, self.middle_landing_spot, self.n_intermediate_path)).astype(int)
            return t if len(my_path) == 0 else np.concatenate((my_path[:-1, :], t))

        path[1] = path[1]
        return self.search_path(path, landing_spot,
                                np.round(np.linspace(initial_pos, path, self.n_intermediate_path)).astype(int))

    def get_distance_to_path(self, new_pos: np.ndarray, path_to_the_landing_spot: np.ndarray) -> float:
        highest = None

        for i, point in enumerate(path_to_the_landing_spot):
            if new_pos[1] >= point[1] and not distance_2(new_pos, point) < 25 ** 2:
            # if new_pos[1] >= point[1]:
                highest = point
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


def norm_reward(feature: float, interval_low: float, interval_high: float) -> float:
    feature = np.clip(feature, interval_low, interval_high)
    return 1.0 - ((feature - interval_low) / (interval_high - interval_low))


n = int(input())
surface = []
for i in range(n):
    land_x, land_y = [int(j) for j in input().split()]
    surface.append([land_x, land_y])

surface = np.array(surface, dtype=int)
i2 = 0
env = None
my_GA = None

while True:
    state = [int(i) for i in input().split()]
    if i2 == 0:
        env = RocketLandingEnv(state, surface)
        my_GA = GeneticAlgorithm(env)
    my_GA.learn(84)
    i2 += 1
