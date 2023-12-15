import random
import time
import timeit

import numpy as np

from src.GA.create_environment import RocketLandingEnv

population_size = 20
horizon = 15


class GeneticAlgorithm:
    def __init__(self, env):
        self.env: RocketLandingEnv = env
        self.population: list[list[list[int]]] = self.init_population()

    def crossover(self, population_survivors: list[list[list[int]]]):
        offspring = []
        while len(offspring) < population_size - len(population_survivors):
            # parent1 = random.choice(population_survivors)
            # parent2 = random.choice(population_survivors)
            parent1 = population_survivors[0]
            parent2 = population_survivors[1]
            policy = []
            for i in range(horizon):
                offspring_rotation = random.randint(min(parent1[i][0], parent2[i][0]),
                                                    max(parent1[i][0], parent2[i][0]))
                offspring_thrust = random.randint(min(parent1[i][1], parent2[i][1]),
                                                  max(parent1[i][1], parent2[i][1]))
                policy.append([offspring_rotation, offspring_thrust])
            offspring.append(policy)
        # offspring.extend(population_survivors)
        return offspring

    def mutation(self, population: list[list[list]]):
        return population

    def heuristic(self, population: list[list[list]], side_values, dist_landing_spot):
        p1 = side_values[0]
        p2 = side_values[1]
        print(dist_landing_spot)
        if dist_landing_spot < 1000:
            for individual in population:
                for action in individual:
                    action[0] = 0
        elif p1 == p2:
            if p1 == 1:
                for individual in population:
                    for action in individual:
                        action[0] -= 5  # check limit -90
            elif p1 == -1:
                for individual in population:
                    for action in individual:
                        action[0] += 5  # check limit 90
        return population

    def roulette_wheel_selection(self, population, fitness_values):
        total_fitness = sum(fitness_values)
        selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
        selected_index = np.random.choice(len(population), p=selection_probabilities)
        return population[selected_index]

    def learn(self, time_available):
        start_time = time.time()
        curr_initial_state = self.env.initial_state
        done = False

        while not done:
            self.env.initial_state = curr_initial_state
            self.population = self.init_population()
            start_time = time.time()
            while (time.time() - start_time) * 1000 < time_available:
                rewards = [self.rollout(individual) for individual in self.population]
                rewards, side, dist_landing_spot = zip(*rewards)
                side = np.array(side)
                dist_landing_spot = np.array(dist_landing_spot)
                self.population = np.array(self.population)
                sorted_indices = np.argsort(rewards)
                parents = self.population[sorted_indices[-2:]]
                selected_side_values = side[sorted_indices[-2:]]
                selected_dist_landing_values = dist_landing_spot[sorted_indices[-2:]]
                self.population = self.crossover(parents)
                self.population = self.heuristic(self.population, selected_side_values, curr_initial_state[7])
                self.population.extend(parents)
                print("Time spent:", (time.time() - start_time) * 1000, "milliseconds")
            rewards = [self.rollout(individual) for individual in self.population]
            rewards, _, _ = zip(*rewards)
            best_individual = self.population[np.argmax(rewards)]
            self.env.reset()
            next_state, _, done, _, _ = self.env.step(best_individual[0])
            curr_initial_state = self.env.denormalize_state(next_state)
        return None

    def rollout(self, policy: list[list[int]]) -> tuple[int, int, float]:
        state, _ = self.env.reset()
        total_reward = 0
        infos = None
        next_state = None
        # i = 0
        # while True:
        for i in range(horizon):
            action = policy[i]
            next_state, reward, done, _, infos = self.env.step(action)
            # self.env.trajectory_plot.append(self.env.denormalize_state(next_state))
            # self.env.rewards_episode.append(reward)
            total_reward += reward  # TODO discounted rewards
            if done:
                break
            i += 1
        self.env.render()
        return total_reward, infos['side'], next_state[7]

    def init_population(self) -> list[list[list[int]]]:
        population = []
        random_action = [0, 0]
        for _ in range(population_size):
            individual = []
            for _ in range(horizon):
                random_action = self.env.generate_random_action(random_action[0], random_action[1])
                individual.append(random_action)
            population.append(individual)
        return population
