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
        offspring.extend(population_survivors)
        return offspring

    def mutation(self, population: list[list[list]]):
        return population

    def heuristic(self, population: list[list[list]]):
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
            while (time.time() - start_time) * 1000 < time_available:
                rewards = np.array([self.rollout(individual) for individual in self.population])
                self.population = np.array(self.population)
                parents = self.population[np.argsort(rewards)[-2:]]

                self.population = self.crossover(parents)
                crossover_time = timeit.timeit(lambda: self.crossover(parents), number=1) * 1000

                # Print the execution time
                print(f"Crossover time: {crossover_time} seconds")
                self.population = self.heuristic(self.population)
                print("Time spent:", (time.time() - start_time) * 1000, "milliseconds")
            rewards = [self.rollout(individual) for individual in self.population]
            best_individual = self.population[np.argmax(rewards)]
            self.env.reset()
            next_state, _, done, _, _ = self.env.step(best_individual[0])
            curr_initial_state = self.env.denormalize_state(next_state)
        return None

    def rollout(self, policy: list[list[int]]) -> float:
        state, _ = self.env.reset()
        total_reward = 0
        # i = 0
        # while True:
        for i in range(horizon):
            action = policy[i]
            next_state, reward, done, _, _ = self.env.step(action)
            # self.env.trajectory_plot.append(self.env.denormalize_state(next_state))
            # self.env.rewards_episode.append(reward)
            total_reward += reward  # TODO discounted rewards
            if done:
                break
            i += 1
        self.env.render()
        return total_reward

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
