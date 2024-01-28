import time
import numpy as np
from nptyping import NDArray, Int, Shape
from source_code.RocketLandingEnv import RocketLandingEnv


class GeneticAlgorithm:
    def __init__(self, env):
        self.env: RocketLandingEnv = env
        self.horizon = 30
        self.offspring_size = 10
        self.n_elites = 4
        self.n_heuristic_guides = 3
        self.mutation_rate = 0.4
        self.population_size = self.offspring_size + self.n_elites + self.n_heuristic_guides
        self.population = self.init_population(self.env.initial_state)

    def crossover(self, population_survivors: np.ndarray, offspring_size):
        offspring = []
        while len(offspring) < offspring_size:
            indices = np.random.randint(population_survivors.shape[0], size=2)
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

    @staticmethod
    def mutation(population, mutation_rate):
        individual = population[np.random.randint(population.shape[0])]
        for action in individual:
            action[1] = 4
        for individual in population:
            for action in individual:
                if np.random.rand() < mutation_rate:
                    action[0] += np.random.randint(-15, 16)
                    action[1] += np.random.randint(-1, 2)
                    action[0] = np.clip(action[0], -90, 90)
                    action[1] = np.clip(action[1], 0, 4)
        return population

    def heuristic(self, curr_initial_state):
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

    def learn(self, time_available_in_ms: int):
        policy = []
        parents = None
        terminated = False
        truncated = False
        while not (terminated or truncated):
            self.population = self.init_population(self.env.initial_state, parents)
            start_time = time.time()
            while (time.time() - start_time) * 1000 < time_available_in_ms:
                rewards = np.array([self.rollout(individual) for individual in self.population])
                self.env.render()
                parents = self.selection(rewards)
                heuristic_guides = self.heuristic(self.env.initial_state)
                heuristic_guides = np.array(
                    [item for item in heuristic_guides if not np.any(np.all(item == parents, axis=(1, 2)))])
                offspring_size = self.offspring_size + self.n_heuristic_guides - len(heuristic_guides)
                offspring = self.crossover(parents, offspring_size)
                offspring = self.mutation(offspring, self.mutation_rate)
                offspring = self.mutation_heuristic(offspring, self.env.initial_state[7])
                self.population = np.concatenate((offspring, parents, heuristic_guides)) if len(
                    heuristic_guides) > 0 else np.concatenate((offspring, parents))
            best_individual = parents[-1]
            self.env.reset()
            action_to_do = self.final_heuristic_verification(best_individual[0], self.env.initial_state)
            next_state, _, terminated, truncated, _ = self.env.step(action_to_do)
            self.env.initial_state = next_state
            parents = parents[:, 1:, :]
            last_elements_tuple = parents[:, -1, :]
            parents = np.concatenate(
                (parents, np.array([self.env.generate_random_action(*item) for item in last_elements_tuple])[:, np.newaxis, :]),
                axis=1)
            policy.append(list(best_individual[0]))
        print(policy)

    def rollout(self, policy) -> float:
        self.env.reset()
        reward = 0

        for action in policy:
            _, reward, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                break
        return reward

    def init_population(self, initial_state, parents=None) -> np.ndarray:
        initial_rotation = initial_state[5]
        initial_thrust = initial_state[6]
        population = np.zeros((self.population_size, self.horizon, 2), dtype=int)
        num_parents = None
        if parents is not None:
            num_parents = len(parents)
            population[:num_parents] = parents
        for i in range(num_parents if parents is not None else 0, self.population_size):
            population[i] = self.generate_random_individual(initial_rotation, initial_thrust)
        return population

    def generate_random_individual(self, previous_rotation: int, previous_thrust: int) -> np.ndarray:
        individual = []
        for _ in range(self.horizon):
            random_action = self.env.generate_random_action(previous_rotation, previous_thrust)
            individual.append(random_action)
            previous_rotation = random_action[0]
            previous_thrust = random_action[1]
        return np.array(individual)

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
    def final_heuristic_verification(action_to_do: np.ndarray[int, 1], state: np.ndarray) -> np.ndarray[int, 1]:
        rotation = state[5]
        if abs(rotation - action_to_do[0]) > 110:
            action_to_do[1] = 0
        return action_to_do

    def selection(self, rewards):
        sorted_indices = np.argsort(rewards)
        parents = self.population[sorted_indices[-self.n_elites:]]
        return parents


def my_random_int(a, b):
    if a == b:
        return a
    else:
        return np.random.randint(min(a, b), max(a, b))
