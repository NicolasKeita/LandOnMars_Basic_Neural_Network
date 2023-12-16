import random
import time

import numpy as np

from src.GA.create_environment import RocketLandingEnv

population_size = 8
offspring_size = population_size - 3
horizon = 50
n_elites = 4


class GeneticAlgorithm:
    def __init__(self, env):
        self.env: RocketLandingEnv = env
        self.population: list[list[list[int]]] = self.init_population(self.env.initial_state[5], self.env.initial_state[6])

    def crossover(self, population_survivors: np.ndarray):
        offspring = []
        while len(offspring) < offspring_size:
            parent1 = population_survivors[np.random.randint(population_survivors.shape[0])]
            parent2 = population_survivors[np.random.randint(population_survivors.shape[0])]
            # parent1 = population_survivors[0]
            # parent2 = population_survivors[1]
            policy = []
            for i in range(horizon):
                offspring_rotation = random.randint(min(parent1[i][0], parent2[i][0]) - 10,
                                                    max(parent1[i][0], parent2[i][0]) + 10)
                offspring_thrust = random.randint(min(parent1[i][1], parent2[i][1]) - 1,
                                                  max(parent1[i][1], parent2[i][1]) + 1)
                # offspring_rotation = random.randint(min(parent1[i][0], parent2[i][0]),
                #                                     max(parent1[i][0], parent2[i][0]))
                # offspring_thrust = random.randint(min(parent1[i][1], parent2[i][1]),
                #                                   max(parent1[i][1], parent2[i][1]))
                offspring_rotation = np.clip(offspring_rotation, -90, 90)
                offspring_thrust = np.clip(offspring_thrust, 0, 4)
                policy.append([offspring_rotation, offspring_thrust])
            offspring.append(policy)
        return np.array(offspring)

    def mutation(self, population: list[list[list]]):
        return population

    def heuristic(self, population: np.ndarray, side_values, dist_landing_spot):
        p1 = side_values[0]
        p2 = side_values[1]
        print("dist landing spot:", dist_landing_spot)

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
        if dist_landing_spot < 300 ** 2:
            individual = population[np.random.randint(population.shape[0])]
            # for individual in population:
            for action in individual:
                action[0] = 0
        population = np.concatenate([np.array([[[0, 4]] * horizon]), population])
        return population

    def learn(self, time_available):
        curr_initial_state = self.env.initial_state
        done = False
        global i2
        i2 = 0
        policy_global = []

        while not done:
            i2 += 1
            self.env.initial_state = curr_initial_state
            self.population = self.init_population(curr_initial_state[5], curr_initial_state[6])
            start_time = time.time()
            while (time.time() - start_time) * 1000 < time_available:
                start_time2 = time.time()
                print("Massive BEGIN rollout")
                rewards = [self.rollout(individual) for individual in self.population]
                print("Choices : ")
                for i in range(len(self.population)):
                    choices = self.population[i]
                    reward = rewards[i][0]
                    print(*choices, reward)
                print(f"Execution time rollout Begin: {(time.time() - start_time2) * 1000} milliseconds")
                rewards, side, _ = zip(*rewards)
                side = np.array(side)
                self.population = np.array(self.population)
                sorted_indices = np.argsort(rewards)
                parents = self.population[sorted_indices[-n_elites:]]
                selected_side_values = side[sorted_indices[-n_elites:]]
                self.population = self.crossover(parents)
                self.population = self.heuristic(self.population, selected_side_values, curr_initial_state[7])
                self.population = np.concatenate((self.population, parents))
                print("Parents1 kept:", *parents[0])
                print("Parents2 kept:", *parents[1])
                print("Parents1 kept:", *parents[2])
                print("Parents2 kept:", *parents[3])
                print("Population after Heuristic and Concatenate: ")
                for i in range(len(self.population)):
                    choices = self.population[i]
                    print(*choices)

                print("Time spent:", (time.time() - start_time) * 1000, "milliseconds")
            print("Massive END rollout")
            rewards = [self.rollout(individual) for individual in self.population]
            rewards, _, _ = zip(*rewards)
            best_individual = self.population[np.argmax(rewards)]
            self.env.reset()
            next_state, _, done, _, _ = self.env.step(best_individual[0])
            print("Action chosen : ", best_individual[0], "Here i2", i2)
            print('State after this action:', self.env.denormalize_state(next_state))
            policy_global.append(list(best_individual[0]))
            curr_initial_state = self.env.denormalize_state(next_state)
        print(policy_global)
        return None

    def rollout(self, policy: list[list[int]]) -> tuple[int, int, float]:
        self.env.reset()
        total_reward = 0
        infos = None
        next_state = None
        start_time = time.time()
        for i in range(horizon):
            action = policy[i]
            next_state, reward, terminated, truncated, infos = self.env.step(action)
            if terminated:
                total_reward = reward
                break
            if truncated:
                total_reward += reward
                break
            else:
                total_reward += reward  # TODO discounted rewards
            i += 1

        execution_time_ms = (time.time() - start_time) * 1000
        print(f" Entire Rollout 50 horizon Execution time: {execution_time_ms} milliseconds")

        self.env.render()
        return total_reward, infos['side'], next_state[7]

    def init_population(self, previous_rotation, previous_thrust) -> list[list[list[int]]]:
        population = []
        for _ in range(population_size):
            individual = []
            random_action = [previous_rotation, previous_thrust]
            for _ in range(horizon):
                random_action = self.env.generate_random_action(random_action[0], random_action[1])
                individual.append(random_action)
            population.append(individual)
        return population
