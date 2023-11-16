import random
import copy

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src.Point2D import Point2D
from src.create_environment import create_env, get_landing_spot_distance, RocketLandingEnv

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


class PolicyNetwork(tf.keras.Model):
    def __init__(self, action_space_size):
        super(PolicyNetwork, self).__init__()
        self.action_space_size = action_space_size

        # Define a simple neural network architecture
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(self.action_space_size, activation='softmax')

    def call(self, state):
        # Forward pass through the neural network to get action probabilities
        x = self.dense1(state)
        x = self.dense2(x)
        action_probabilities = self.output_layer(x)
        return action_probabilities


class ParticleSwarmOptimization:
    def __init__(self, env):  # TODO add hyperparams to constructor params
        self.env: RocketLandingEnv = env # TODO remove type

        self.global_best_value = None
        self.global_best_position = None
        self.personal_best_positions = None

        self.personal_best_values = None
        self.particles_velocity = None
        self.particles_position = None

        self.num_particles = 20
        self.num_dimensions = 6
        self.num_iterations = 100
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5

        self.personal_weight = 1.5
        self.global_weight = 1.5

        self.horizon_size = 80 # TODO change to 700

    def run(self):
        self.initialize_population()
        policy_network = PolicyNetwork(720)
        for _ in range(self.num_iterations):
            # Evaluate fitness
            # fitness = fitness_function(self.particles_position[i], grid, landing_spot, 5500)

            # Evaluate fitness for each particle
            fitness_values = np.zeros(self.num_particles)
            for particle_index in range(self.num_particles):
                weights = self.particles_position[particle_index, :]
                print(weights)
                exit(3)
                policy_network.set_weights(weights)
                total_reward = evaluate_policy(policy_network, self.env)
                fitness_values[particle_index] = total_reward

                # Update personal best
                # if fitness < self.personal_best_values[i]:
                #     self.personal_best_values[i] = fitness
                #     self.personal_best_positions[i] = self.particles_position[i].copy()

                # Update personal best if better
                if total_reward > fitness_values[particle_index]:
                    self.personal_best_positions[particle_index, :] = self.particles_position[particle_index, :]

            # Find the global best particle
            global_best_index = np.argmax(fitness_values)
            global_best_position = self.particles_position[global_best_index, :]

            #     # Update global best
            # if fitness < self.global_best_value:
            #     self.global_best_value = fitness
            #     self.global_best_position = self.particles_position[i].copy()

            # # Update particle velocities and positions
            # for i in range(self.num_particles):
            #     r1, r2 = np.random.rand(), np.random.rand()
            #     cognitive_term = self.cognitive_param * r1 * (self.personal_best_positions[i] - self.particles_position[i])
            #     social_term = self.social_param * r2 * (self.global_best_position - self.particles_position[i])
            #     self.particles_velocity[i] = self.inertia_weight * self.particles_velocity[i] + cognitive_term + social_term
            #     self.particles_position[i] += self.particles_velocity[i]

            # Update particle velocities and positions
            self.inertia_term = self.inertia_weight * self.particles_velocity
            self.personal_term = self.personal_weight * np.random.rand() * (
                        self.personal_best_positions - self.particles_position)
            self.global_term = self.global_weight * np.random.rand() * (global_best_position - self.particles_position)

            self.particles_velocity = self.inertia_term + self.personal_term + self.global_term
            self.particles_position = self.particles_position + self.particles_velocity

        best_weights = self.personal_best_positions[np.argmax(fitness_values), :]
        policy_network.set_weights(best_weights)
        return
        # return self.global_best_position, self.global_best_value

    def initialize_population(self):
        population = []
        for _ in range(self.num_particles):
            previous_action = [0, 0]
            policy = []
            for i in range(self.horizon_size):
                random_action_index, random_action = self.env.generate_random_action(previous_action[0], previous_action[1])
                previous_action[0] = random_action[0]
                previous_action[1] = random_action[1]
                policy.append(random_action_index)
            population.append(policy)

        velocities = []
        for _ in range(self.num_particles):
            velocity = []
            for _ in range(self.horizon_size):
                velocity.append((random.randint(-5, 5), random.randint(-1, 1)))
            velocities.append(velocity)

        self.personal_best_positions = copy.deepcopy(population)
        exit(55)
        # self.particles_position = np.random.rand(self.num_particles, self.num_dimensions) * 10
        # self.particles_velocity = np.random.rand(self.num_particles, self.num_dimensions)
        # self.personal_best_positions = self.particles_position.copy()

        # self.personal_best_values = np.array([float('inf')] * self.num_particles)
        # self.global_best_position = np.zeros(self.num_dimensions)
        # self.global_best_value = float('inf')


# Function to evaluate a policy in the RL environment
def evaluate_policy(policy_network, env):
    # Implement the logic to run the policy in the environment and return the total reward
    # Use policy_network to get actions based on states and interact with the environment
    total_reward = 0
    # ...

    return total_reward
