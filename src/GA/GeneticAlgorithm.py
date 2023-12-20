import time
import numpy as np

from src.GA.create_environment import RocketLandingEnv


class GeneticAlgorithm:
    def __init__(self, env):
        self.env: RocketLandingEnv = env

        self.horizon = 20
        self.offspring_size = 10
        self.n_elites = 4
        self.n_heuristic_guides = 3
        self.population_size = self.offspring_size + self.n_elites + self.n_heuristic_guides

        self.population = self.init_population(self.env.initial_state[5], self.env.initial_state[6])

    def crossover(self, population_survivors: np.ndarray, offspring_size):
        offspring = []
        while len(offspring) < offspring_size:
            indices = np.random.randint(population_survivors.shape[0], size=(2))
            parents = population_survivors[indices]
            policy = np.zeros((self.horizon, 2), dtype=int)
            for i in range(self.horizon):
                # offspring_rotation = np.random.randint(np.amin(parents[:, i, 0], axis=0) - 10,
                #                                        np.amax(parents[:, i, 0], axis=0) + 10)
                # offspring_thrust = np.random.randint(np.amin(parents[:, i, 1], axis=0) - 1,
                #                                      np.amax(parents[:, i, 1], axis=0) + 1)
                offspring_rotation = my_random_int(parents[0, i, 0], parents[1, i, 0])
                offspring_thrust = my_random_int(parents[0, i, 1], parents[1, i, 1])
                offspring_rotation = np.clip(offspring_rotation, -90, 90)
                offspring_thrust = np.clip(offspring_thrust, 0, 4)
                policy[i] = [offspring_rotation, offspring_thrust]
            offspring.append(policy)
        return np.array(offspring)

    def mutation(self, population: np.ndarray):
        mutation_rate = 0.1

        individual = population[np.random.randint(population.shape[0])]
        for action in individual:
            action[1] = 4
        for individual in population:
        # individual = population[np.random.randint(population.shape[0])]
            for action in individual:
                if np.random.rand() < mutation_rate:
                    action[0] += np.random.randint(-5, 6)
                    action[1] += np.random.randint(-1, 2)
                    action[0] = np.clip(action[0], -90, 90)
                    action[1] = np.clip(action[1], 0, 4)
        return population

    def heuristic(self, curr_initial_state):
        heuristics_guides = np.zeros((3, self.horizon, 2), dtype=int)
        # one guide to slow down full speed, one guide to go toward the target full speed

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

        # if dist_landing_spot < 300 ** 2:
        #     policy = np.empty((50, 2))
        #     for action in policy:
        #         action[0] = 0
        return heuristics_guides

    def learn(self, time_available):
        curr_initial_state = self.env.initial_state
        terminated = False
        truncated = False
        global i2
        i2 = 0
        policy_global = []
        parents = None

        # _, unique_indices = np.unique(self.population, axis=0, return_index=True)
        # if len(unique_indices) != self.population.shape[0]:
        #     raise ValueError("Duplicate individuals found in the population.")
        # population = np.concatenate((self.population, heuristic_guides))
        while not (terminated or truncated):
            i2 += 1
            self.env.initial_state = curr_initial_state
            self.population = self.init_population(curr_initial_state[5], curr_initial_state[6], parents)
            start_time = time.time()
            while True:
                start_time_2 = time.time()

                rewards = [self.rollout(individual) for individual in self.population]
                rewards, rewards_direct = zip(*rewards)
                sorted_indices = np.argsort(rewards)
                print("sorted indices : ", sorted_indices)
                for i, indiv in enumerate(self.population):
                    print("pop", *indiv, rewards[i], "direct:", rewards_direct[i])
                parents = self.population[sorted_indices[-self.n_elites:]]
                for indiv in parents:
                    print("parents", *indiv)


                if (time.time() - start_time) * 1000 >= time_available:
                    break
                # side = np.array(side)
                # selected_side_values = side[sorted_indices[-self.n_elites:]]

                heuristic_guides = self.heuristic(curr_initial_state)
                # for indiv in heuristic_guides:
                #     print("heuristic", *indiv)
                heuristic_guides = np.array(
                    [item for item in heuristic_guides if not np.any(np.all(item == parents, axis=(1, 2)))])
                # for indiv in heuristic_guides:
                #     print("heuritstic2", *indiv)
                offspring_size = self.offspring_size + self.n_heuristic_guides - len(heuristic_guides)
                offspring = self.crossover(parents, offspring_size)
                offspring = self.mutation(offspring)
                offspring = self.mutation_heuristic(offspring, curr_initial_state[7])
                # for i, indiv in enumerate(self.population):
                #     print("after crossover", *indiv)
                # self.population = np.concatenate((offspring, parents, heuristic_guides))
                self.population = np.concatenate((offspring, parents, heuristic_guides)) if len(
                    heuristic_guides) > 0 else np.concatenate((offspring, parents))
                for i, indiv in enumerate(self.population):
                    print("after concatenate", *indiv)
                # self.population = self.replace_duplicates_with_random(self.population, curr_initial_state[5],
                #                                                       curr_initial_state[6])
                print("One generation duration:", (time.time() - start_time_2) * 1000, "milliseconds")
            best_individual = parents[-1]
            self.env.reset()
            action_to_do = self.final_heuristic_verification(best_individual[0], curr_initial_state)
            next_state, _, terminated, truncated, _ = self.env.step(action_to_do)
            print("Action chosen : ", action_to_do, "Here i2", i2)
            # print('State after this action:', next_state)
            policy_global.append(list(best_individual[0]))
            curr_initial_state = next_state
            parents = parents[:, 1:, :]
            last_elements_tuple = parents[:, -1, :]
            parents = np.concatenate(
                (parents, np.array([self.env.generate_random_action(*item) for item in last_elements_tuple])[:, np.newaxis, :]),
                axis=1)
        print(policy_global)
        return None

    def rollout(self, policy: np.ndarray) -> tuple[int, int]:
        self.env.reset()
        total_reward = 0
        infos = None
        first_reward = None
        discount_factor = 0.91

        for i, action in enumerate(policy):
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            if i == 1:
                first_reward = reward
            if terminated:
                total_reward = reward
                break
            if truncated:
                # total_reward += reward * (discount_factor ** i)
                total_reward = reward
                # print("ONE DONE", total_reward)
                break
            else:
                total_reward = reward
                # total_reward += reward * (discount_factor ** i)
            i += 1
        self.env.render()
        return total_reward, first_reward

    def init_population(self, previous_rotation, previous_thrust, parents=None) -> np.ndarray:
        population = np.zeros((self.population_size, self.horizon, 2), dtype=int)
        num_parents = None
        if parents is not None:
            num_parents = len(parents)
            population[:num_parents] = parents
        for i in range(num_parents if parents is not None else 0, self.population_size):
            population[i] = self.generate_random_individual(previous_rotation, previous_thrust)
        return population

    def generate_random_individual(self, previous_rotation, previous_thrust) -> np.ndarray:
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

    def final_heuristic_verification(self, action_to_do, state):
        rotation = state[5]
        if abs(rotation - action_to_do[0]) > 110:
            action_to_do[1] = 0
        # else: #TODO remove
        #     action_to_do[1] = 4#TODO remove
        return action_to_do


def my_random_int(a, b):
    if a == b:
        return a
    else:
        return np.random.randint(min(a, b), max(a, b))
