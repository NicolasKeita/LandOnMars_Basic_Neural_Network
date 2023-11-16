import numpy as np

from src.Point2D import Point2D
from src.create_environment import create_env, get_landing_spot_distance

mars_surface = [Point2D(0, 100), Point2D(1000, 500), Point2D(1500, 1500), Point2D(3000, 1000), Point2D(4000, 150), Point2D(5500, 150), Point2D(6999, 800)]
grid: list[list[bool]] = create_env(mars_surface, 7000, 3000)
landing_spot = (Point2D(4000, 150), Point2D(5500, 150))


def fitness_normalize_unsuccessful_rewards(state, landing_spot):
    rocket_pos_x = round(state[0])
    hs = state[2]
    vs = state[3]
    rotation = state[5]
    dist = get_landing_spot_distance(rocket_pos_x, landing_spot[0].x, landing_spot[1].x)
    norm_dist = 1.0 if dist > 2000 else dist / 2000
    print("crash", dist)
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


class ParticleSwarmOptimization:
    def __init__(self):  # TODO add hyperparams to constructor params
        self.global_best_value = None
        self.global_best_position = None
        self.personal_best_values = None
        self.personal_best_positions = None
        self.particles_velocity = None
        self.particles_position = None

        self.num_particles = 20
        self.num_dimensions = 6
        self.num_iterations = 100
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5

    def run(self):
        self.initialize_particles()
        for _ in range(self.num_iterations):
            for i in range(self.num_particles):
                # Evaluate fitness
                fitness = fitness_function(self.particles_position[i], grid, landing_spot, 5500)

                # Update personal best
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.particles_position[i].copy()

                # Update global best
                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.particles_position[i].copy()

            # Update particle velocities and positions
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_term = self.cognitive_param * r1 * (self.personal_best_positions[i] - self.particles_position[i])
                social_term = self.social_param * r2 * (self.global_best_position - self.particles_position[i])
                self.particles_velocity[i] = self.inertia_weight * self.particles_velocity[i] + cognitive_term + social_term
                self.particles_position[i] += self.particles_velocity[i]
        return self.global_best_position, self.global_best_value

    def initialize_particles(self):
        self.particles_position = np.random.rand(self.num_particles, self.num_dimensions) * 10
        self.particles_velocity = np.random.rand(self.num_particles, self.num_dimensions)
        self.personal_best_positions = self.particles_position.copy()
        self.personal_best_values = np.array([float('inf')] * self.num_particles)
        self.global_best_position = np.zeros(self.num_dimensions)
        self.global_best_value = float('inf')
