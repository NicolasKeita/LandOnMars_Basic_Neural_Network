import time
import numpy as np

from src.GA.create_environment import RocketLandingEnv


offspring_size = 3
horizon = 50
n_elites = 4
population_size = offspring_size + n_elites + 1


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
        heuristics_guides = np.zeros((2, horizon, 2), dtype=int)

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

        x = curr_initial_state[0]
        y = curr_initial_state[1]
        goal = np.mean(self.env.landing_spot, axis=1)
        angle = np.arctan2(goal[1] - y, goal[0] - x)
        angle_degrees = np.degrees(angle)
        for action in heuristics_guides[0]:
            action[1] = 4
            action[0] = np.clip(round(angle_degrees), -90, 90)

        if dist_landing_spot < 300 ** 2:
            individual = population[np.random.randint(population.shape[0])]
            for action in individual:
                action[0] = 0
        population = np.concatenate([np.array([[[0, 4]] * horizon]), population])
        return population, heuristics_guides

    def learn(self, time_available):
        curr_initial_state = self.env.initial_state
        terminated = False
        truncated = False
        global i2
        i2 = 0
        policy_global = []
        parents = None

        while not (terminated or truncated):
            i2 += 1
            self.env.initial_state = curr_initial_state
            self.population = self.init_population(curr_initial_state[5], curr_initial_state[6], parents)
            start_time = time.time()
            while True:
                start_time_2 = time.time()
                # _, unique_indices = np.unique(self.population, axis=0, return_index=True)
                # if len(unique_indices) != self.population.shape[0]:
                #     raise ValueError("Duplicate individuals found in the population.")

                rewards = [self.rollout(individual) for individual in self.population]
                rewards, side = zip(*rewards)
                sorted_indices = np.argsort(rewards)
                parents = self.population[sorted_indices[-n_elites:]]
                if (time.time() - start_time) * 1000 >= time_available:
                    break
                side = np.array(side)
                selected_side_values = side[sorted_indices[-n_elites:]]
                self.population = self.crossover(parents)
                self.population = self.replace_duplicates_with_random(self.population, curr_initial_state[5],
                                                                      curr_initial_state[6])
                self.population, heuristic_guides = self.heuristic(self.population, selected_side_values, curr_initial_state[7], curr_initial_state)
                self.population = self.replace_duplicates_with_random(self.population, curr_initial_state[5],
                                                                      curr_initial_state[6])
                self.population = np.concatenate((self.population, parents, heuristic_guides))
                self.population = self.replace_duplicates_with_random(self.population, curr_initial_state[5],
                                                                      curr_initial_state[6])
                print("One generation duration:", (time.time() - start_time_2) * 1000, "milliseconds")
            best_individual = parents[-1]
            self.env.reset()
            next_state, _, terminated, truncated, _ = self.env.step(best_individual[0])
            print("Action chosen : ", best_individual[0], "Here i2", i2)
            # print('State after this action:', next_state)
            policy_global.append(list(best_individual[0]))
            curr_initial_state = next_state
        print(policy_global)
        return None

    def rollout(self, policy: np.ndarray) -> tuple[int, int]:
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
        return total_reward, infos['side']

    def init_population(self, previous_rotation, previous_thrust, parents=None) -> np.ndarray:
        population = np.zeros((population_size, horizon, 2), dtype=int)
        num_parents = None
        if parents is not None:
            num_parents = len(parents)
            population[:num_parents] = parents
        for i in range(num_parents if parents is not None else 0, population_size):
            population[i] = self.generate_random_individual(previous_rotation, previous_thrust)
        return population

    def generate_random_individual(self, previous_rotation, previous_thrust) -> np.ndarray:
        individual = np.zeros((horizon, 2), dtype=int)
        random_action = np.array([previous_rotation, previous_thrust])
        for i in range(horizon):
            random_action = self.env.generate_random_action(random_action[0], random_action[1])
            individual[i] = random_action
        return individual

    def replace_duplicates_with_random(self, population, previous_rotation, previous_thrust):
        _, unique_indices = np.unique(population, axis=0, return_index=True)
        duplicate_indices = np.setdiff1d(np.arange(population.shape[0]), unique_indices)
        for i in duplicate_indices:
            population[i] = self.generate_random_individual(previous_rotation, previous_thrust)
        return population
