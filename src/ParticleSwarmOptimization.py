import random
import copy

import numpy as np

from src.Point2D import Point2D
from src.create_environment import create_env, get_landing_spot_distance, RocketLandingEnv
from src.graph_handler import display_graph

mars_surface = [Point2D(0, 100), Point2D(1000, 500), Point2D(1500, 1500), Point2D(3000, 1000), Point2D(4000, 150),
                Point2D(5500, 150), Point2D(6999, 800)]
grid: list[list[bool]] = create_env(mars_surface, 7000, 3000)
landing_spot = (Point2D(4000, 150), Point2D(5500, 150))


def fitness_normalize_unsuccessful_rewards(state, landing_spot):
    rocket_pos_x = round(state[0])
    hs = state[2]
    vs = state[3]
    rotation = state[5]
    dist = get_landing_spot_distance(rocket_pos_x, landing_spot[0].x, landing_spot[1].x)
    norm_dist = 1.0 if dist > 2000 else dist / 2000
    # print("crash", dist)
    return norm_dist


def fitness_function(state, grid, landing_spot, initial_fuel) -> float:
    rocket_pos_x = round(state[0])
    rocket_pos_y = round(state[1])
    hs = state[2]
    vs = state[3]
    remaining_fuel = state[4]
    fuel_used_so_far = initial_fuel - remaining_fuel
    rotation = state[5]

    if (landing_spot[0].x <= rocket_pos_x <= landing_spot[1].x and landing_spot[0].y >= rocket_pos_y and
            rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20):
        print("GOOD", rocket_pos_x, remaining_fuel)
        return fuel_used_so_far / initial_fuel
    if (rocket_pos_y < 0 or rocket_pos_y >= 3000 or rocket_pos_x < 0 or rocket_pos_x >= 7000
            or grid[rocket_pos_y][rocket_pos_x] is False or remaining_fuel < -4):
        return fitness_normalize_unsuccessful_rewards(state, landing_spot)
    return 1

#
# class PolicyNetwork(tf.keras.Model):
#     def __init__(self, action_space_size):
#         super(PolicyNetwork, self).__init__()
#         self.action_space_size = action_space_size
#
#         # Define a simple neural network architecture
#         self.dense1 = layers.Dense(64, activation='relu')
#         self.dense2 = layers.Dense(64, activation='relu')
#         self.output_layer = layers.Dense(self.action_space_size, activation='softmax')
#
#     def call(self, state):
#         # Forward pass through the neural network to get action probabilities
#         x = self.dense1(state)
#         x = self.dense2(x)
#         action_probabilities = self.output_layer(x)
#         return action_probabilities


class ParticleSwarmOptimization:
    def __init__(self, env):  # TODO add hyperparams to constructor params
        self.env: RocketLandingEnv = env # TODO remove type

        self.global_best_value = None
        self.global_best_policy = None
        self.personal_best_policies = None

        self.personal_best_values = None
        self.velocities = None
        self.population = None

        self.n_population = 20
        self.num_dimensions = 6
        self.n_episodes = 100
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5

        self.personal_weight = 1.5
        self.global_weight = 1.5

        self.horizon_size = 80  # TODO change to 700

    def run(self):
        self.initialize_population()
        # policy_network = PolicyNetwork(720)
        for episode_index in range(self.n_episodes):
            # Evaluate fitness
            # fitness = fitness_function(self.particles_position[i], grid, landing_spot, 5500)

            # Evaluate fitness for each particle
            fitness_values = np.zeros(self.n_population)
            trajectories = []
            for particle_index in range(self.n_population):
                state_value, trajectory = evaluate_policy(self.env, self.population[particle_index])
                trajectories.append(trajectory)
                fitness_values[particle_index] = state_value

                # Update personal best
                if state_value < self.personal_best_values[particle_index]:
                    self.personal_best_values[particle_index] = state_value
                    self.personal_best_policies[particle_index] = copy.deepcopy(self.population[particle_index])
                #     self.personal_best_positions[i] = self.particles_position[i].copy()

                if state_value < self.global_best_value:
                    self.global_best_value = state_value
                    self.global_best_policy = copy.deepcopy(self.population[particle_index])  # TODO see if i can skip copying

            display_graph(trajectories, episode_index)

            for i, policy_indexes in enumerate(self.population):
                r1, r2 = np.random.rand(), np.random.rand()
                policy = self.env.action_indexes_to_real_action(policy_indexes)
                p_best_policy = self.env.action_indexes_to_real_action(self.personal_best_policies[i])
                g_best_policy = self.env.action_indexes_to_real_action(self.global_best_policy)

                cognitive_term = self.cognitive_param * r1 * (np.array(p_best_policy) - np.array(policy))
                social_term = self.social_param * r2 * (np.array(g_best_policy) - np.array(policy))
                mod_velocities = []
                for velocity in self.velocities[i]:
                    mod_velocities.append((round(velocity[0] * self.inertia_weight), round(velocity[1] * self.inertia_weight)))
                self.velocities[i] = mod_velocities + cognitive_term + social_term
                policy = np.array(policy) + np.array(self.velocities[i])
                self.population[i] = self.env.real_actions_to_indexes(policy)

        return self.global_best_policy, self.global_best_value
            # # Update particle velocities and positions
            # for i in range(self.num_particles):
            #     r1, r2 = np.random.rand(), np.random.rand()
            #     cognitive_term = self.cognitive_param * r1 * (self.personal_best_positions[i] - self.particles_position[i])
            #     social_term = self.social_param * r2 * (self.global_best_position - self.particles_position[i])
            #     self.particles_velocity[i] = self.inertia_weight * self.particles_velocity[i] + cognitive_term + social_term
            #     self.particles_position[i] += self.particles_velocity[i]

            # # Update particle velocities and positions
            # self.inertia_term = self.inertia_weight * self.velocities
            # self.personal_term = self.personal_weight * np.random.rand() * (
            #             self.personal_best_positions - self.population)
            # self.global_term = self.global_weight * np.random.rand() * (global_best_position - self.population)
            #
            # self.velocities = self.inertia_term + self.personal_term + self.global_term
            # self.population = self.population + self.velocities

        # best_weights = self.personal_best_positions[np.argmax(fitness_values), :]
        # policy_network.set_weights(best_weights)
        # return
        # return self.global_best_position, self.global_best_value

    def initialize_population(self):
        population = []
        for _ in range(self.n_population):
            previous_action = [0, 0]
            policy = []
            for i in range(self.horizon_size):
                random_action_index, random_action = self.env.generate_random_action(previous_action[0], previous_action[1])
                previous_action[0] = random_action[0]
                previous_action[1] = random_action[1]
                policy.append(random_action_index)
            population.append(policy)
        self.population = population

        velocities = []
        for _ in range(self.n_population):
            velocity = []
            for _ in range(self.horizon_size):
                velocity.append((random.randint(-5, 5), random.randint(-1, 1)))
            velocities.append(velocity)
        self.velocities = velocities

        self.personal_best_policies = copy.deepcopy(population)
        self.personal_best_values = np.array([float('inf')] * self.n_population)
        # self.particles_position = np.random.rand(self.num_particles, self.num_dimensions) * 10
        # self.particles_velocity = np.random.rand(self.num_particles, self.num_dimensions)
        # self.personal_best_positions = self.particles_position.copy()

        # self.personal_best_values = np.array([float('inf')] * self.num_particles)
        # self.global_best_position = np.zeros(self.num_dimensions)
        self.global_best_value = float('inf')


#  TODO inheritance the class rocketLanding
def evaluate_policy(env: RocketLandingEnv, policy):
    cumulated_reward = 0
    trajectory = []
    trajectory.append((env.state[0], env.state[1]))
    for action in policy:
        next_state, reward, done, _ = env.step(action)
        trajectory.append((next_state[0], next_state[1]))
        cumulated_reward += reward
        if done:
            break
    env.reset()
    return cumulated_reward, trajectory
