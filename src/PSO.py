import numpy as np
import matplotlib.pyplot as plt


# Objective function to be minimized (you can replace it with your own function)
def objective_function(x):
    return x ** 2 - 4 * x + 4


# Particle Swarm Optimization
def particle_swarm_optimization(objective_function, num_particles, num_dimensions, num_iterations, inertia_weight,
                                cognitive_param, social_param):
    # Initialize particles
    particles_position = np.random.rand(num_particles, num_dimensions) * 10
    particles_velocity = np.random.rand(num_particles, num_dimensions)
    personal_best_positions = particles_position.copy()
    personal_best_values = np.array([float('inf')] * num_particles)
    global_best_position = np.zeros(num_dimensions)
    global_best_value = float('inf')

    # Main loop
    for _ in range(num_iterations):
        for i in range(num_particles):
            # Evaluate fitness
            fitness = objective_function(particles_position[i])

            # Update personal best
            if fitness < personal_best_values[i]:
                personal_best_values[i] = fitness
                personal_best_positions[i] = particles_position[i].copy()

            # Update global best
            if fitness < global_best_value:
                global_best_value = fitness
                global_best_position = particles_position[i].copy()

        # Update particle velocities and positions
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_term = cognitive_param * r1 * (personal_best_positions[i] - particles_position[i])
            social_term = social_param * r2 * (global_best_position - particles_position[i])
            particles_velocity[i] = inertia_weight * particles_velocity[i] + cognitive_term + social_term
            particles_position[i] += particles_velocity[i]

    return global_best_position, global_best_value


# Example usage
num_particles = 20
num_dimensions = 1
num_iterations = 100
inertia_weight = 0.7
cognitive_param = 1.5
social_param = 1.5

best_position, best_value = particle_swarm_optimization(objective_function, num_particles, num_dimensions,
                                                        num_iterations, inertia_weight, cognitive_param, social_param)

print(f"Best position found: {best_position}")
print(f"Best value found: {best_value}")

# Plot the objective function
x = np.linspace(-10, 10, 100)
y = objective_function(x)
plt.plot(x, y, label='Objective Function')
plt.scatter(best_position, best_value, color='red', marker='X', label='Global Best')
plt.legend()
plt.title('Particle Swarm Optimization')
plt.xlabel('x')
plt.ylabel('Objective Value')
plt.show()
