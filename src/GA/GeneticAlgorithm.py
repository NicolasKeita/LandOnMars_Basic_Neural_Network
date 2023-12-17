import random
import time

import numpy as np
from shapely import LineString

from src.GA.create_environment import RocketLandingEnv

population_size = 8
offspring_size = population_size - 3
horizon = 30
n_elites = 4


class GeneticAlgorithm:
    def __init__(self, env):
        self.env: RocketLandingEnv = env
        self.population = self.init_population(self.env.initial_state[5], self.env.initial_state[6])

    def crossover(self, population_survivors: np.ndarray):
        offspring = []
        while len(offspring) < offspring_size:
            indices = np.random.randint(population_survivors.shape[0], size=2)
            parents = population_survivors[indices]
            policy = np.zeros((horizon, 2), dtype=int)
            for i in range(horizon):
                offspring_rotation = np.random.randint(np.amin(parents[:, i, 0], axis=0) - 10,
                                                       np.amax(parents[:, i, 0], axis=0) + 10)
                offspring_thrust = np.random.randint(np.amin(parents[:, i, 1], axis=0) - 1,
                                                     np.amax(parents[:, i, 1], axis=0) + 1)
                offspring_rotation = np.clip(offspring_rotation, -90, 90)
                offspring_thrust = np.clip(offspring_thrust, 0, 4)
                policy[i] = [offspring_rotation, offspring_thrust]
            offspring.append(policy)
        return np.array(offspring)

    def mutation(self, population: list[list[list]]):
        return population

    def heuristic(self, population: np.ndarray, side_values, dist_landing_spot, curr_initial_state):
        p1 = side_values[0]
        p2 = side_values[1]

        if p1 == p2:
            if p1 == 1:
                for individual in population:
                    for action in individual:
                        action[0] -= 5
                        action[0] = np.clip(action[0], -90, 90)
            elif p1 == -1:
                for individual in population:
                    for action in individual:
                        action[0] += 5
                        action[0] = np.clip(action[0], -90, 90)

        individual = population[np.random.randint(population.shape[0])]
        x = curr_initial_state[0]
        y = curr_initial_state[1]
        landing_spot = self.env.landing_spot
        if isinstance(self.env.landing_spot, LineString):
            landing_spot = np.array(self.env.landing_spot.xy)
        goal = np.mean(landing_spot, axis=1)
        angle = np.arctan2(goal[1] - y, goal[0] - x)
        angle_degrees = np.degrees(angle)
        for action in individual:
            action[1] = 4
            action[0] = np.clip(round(angle_degrees), -90, 90)

        if dist_landing_spot < 300 ** 2:
            individual = population[np.random.randint(population.shape[0])]
            # for individual in population:
            for action in individual:
                action[0] = 0
            # individual = population[np.random.randint(population.shape[0])]
            # x = curr_initial_state[0]
            # y = curr_initial_state[1]
            # landing_spot = self.env.landing_spot
            # if isinstance(self.env.landing_spot, LineString):
            #     landing_spot = np.array(self.env.landing_spot.xy)
            # goal = np.mean(landing_spot, axis=1)
            # angle = np.arctan2(goal[1] - y, goal[0] - x)
            # angle_degrees = np.degrees(angle)
            # for action in individual:
            #     action[1] = 4
            #     action[0] = round(angle_degrees)
        population = np.concatenate([np.array([[[0, 4]] * horizon]), population])
        return population

    def learn(self, time_available):
        curr_initial_state = self.env.initial_state
        terminated = False
        truncated = False
        global i2
        i2 = 0
        policy_global = []

        while not (terminated or truncated):
            i2 += 1
            self.env.initial_state = curr_initial_state
            self.population = self.init_population(curr_initial_state[5], curr_initial_state[6])
            start_time = time.time()
            while (time.time() - start_time) * 1000 < time_available:
                rewards = [self.rollout(individual) for individual in self.population]
                rewards, side, _ = zip(*rewards)
                sorted_indices = np.argsort(rewards)
                parents = self.population[sorted_indices[-n_elites:]]
                side = np.array(side)
                selected_side_values = side[sorted_indices[-n_elites:]]
                self.population = self.crossover(parents)
                self.population = self.heuristic(self.population, selected_side_values, curr_initial_state[7], curr_initial_state)
                self.population = np.concatenate((self.population, parents))
                # print("Time spent:", (time.time() - start_time) * 1000, "milliseconds")
            # print("Massive END rollout")
            rewards = [self.rollout(individual) for individual in self.population]
            rewards, _, _ = zip(*rewards)
            best_individual = self.population[np.argmax(rewards)]
            self.env.reset()
            next_state, _, terminated, truncated, _ = self.env.step(best_individual[0])
            print("Action chosen : ", best_individual[0], "Here i2", i2)
            # print('State after this action:', next_state)
            policy_global.append(list(best_individual[0]))
            curr_initial_state = next_state
        print(policy_global)
        return None

    def rollout(self, policy: np.ndarray) -> tuple[int, int, float]:
        self.env.reset()
        total_reward = 0
        infos = None
        next_state = None
        discount_factor = 0.91

        for i, action in enumerate(policy):
            next_state, reward, terminated, truncated, infos = self.env.step(action)
            if terminated:
                total_reward = reward
                break
            if truncated:
                total_reward += reward * (discount_factor ** i)
                break
            else:
                total_reward += reward * (discount_factor ** i)
            i += 1
        self.env.render()
        return total_reward, infos['side'], next_state[7]

    # def init_population(self, previous_rotation, previous_thrust) -> list[list[list[int]]]:
    #     population = []
    #     for _ in range(population_size):
    #         individual = []
    #         random_action = [previous_rotation, previous_thrust]
    #         for _ in range(horizon):
    #             random_action = self.env.generate_random_action(random_action[0], random_action[1])
    #             individual.append(random_action)
    #         population.append(individual)
    #     return population

    def init_population(self, previous_rotation, previous_thrust) -> np.ndarray:
        population = np.empty((population_size, horizon, 2), dtype=int)

        for i in range(population_size):
            random_action = np.array([previous_rotation, previous_thrust])
            for j in range(horizon):
                random_action = self.env.generate_random_action(random_action[0], random_action[1])
                population[i, j] = random_action

        return population
