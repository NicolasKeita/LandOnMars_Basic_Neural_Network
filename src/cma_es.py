import cma
import numpy as np

from src.Point2D import Point2D
from src.create_environment import normalize_unsuccessful_rewards, create_env

mars_surface = [(0, 100), (1000, 500), (1500, 1500), (3000, 1000), (4000, 150), (5500, 150), (6999, 800)]
mars_surface = [Point2D(0, 100), Point2D(1000, 500), Point2D(1500, 1500), Point2D(3000, 1000), Point2D(4000, 150), Point2D(5500, 150), Point2D(6999, 800)]
grid: list[list[bool]] = create_env(mars_surface, 7000, 3000)
landing_spot = [(4000, 150), (5500, 150)]
landing_spot = (Point2D(4000, 150), Point2D(5500, 150))

def fitness_function(state, grid, landing_spot) -> float:
    rocket_pos_x = round(state[0])
    rocket_pos_y = round(state[1])
    hs = state[2]
    vs = state[3]
    remaining_fuel = state[4]
    rotation = state[5]

    if (landing_spot[0].x <= rocket_pos_x <= landing_spot[1].x and landing_spot[0].y >= rocket_pos_y and
            rotation == 0 and abs(vs) <= 40 and abs(hs) <= 20):
        print("GOOD", rocket_pos_x, remaining_fuel)
        return remaining_fuel * 10
    if (rocket_pos_y < 0 or rocket_pos_y >= 3000 or rocket_pos_x < 0 or rocket_pos_x >= 7000
            or grid[rocket_pos_y][rocket_pos_x] is False or remaining_fuel < -4):
        return normalize_unsuccessful_rewards(state, landing_spot)
    return 0

# Define the Sphere function (objective function to be minimized)
def sphere(x):
    return sum(x ** 2)


target_position = [5, 5]


# def fitness_function(state):
#     print(state)
#     # Assuming state is a vector with position, speed, rotation, and power
#     position = state[0:2]
#     speed = state[2:4]
#     rotation = state[4]
#     power = state[5]
#
#     # Example fitness calculation (you should replace this with your own logic)
#     fitness = -((position[0] - target_position[0]) ** 2 + (position[1] - target_position[1]) ** 2)
#
#     # You can incorporate other factors like speed, rotation, power, etc., based on your problem
#
#     return fitness


# Set up the CMA-ES optimizer
# initial_mean = np.zeros(6)  # Initial guess for the mean vector
#
# print(initial_mean)
# exit(0)
# initial_mean = []
initial_mean = np.array([2500, 2700, 0, 0, 550, 0, 0])
sigma = 0.5  # Initial step size (standard deviation)
options = {'maxiter': 1000, 'tolfun': 1e-6}  # Maximum number of iterations and convergence criterion
es = cma.CMAEvolutionStrategy(initial_mean, sigma, options)

# Run the optimization loop
while not es.stop():
    solutions = es.ask()  # Sample new solutions
    fitness_values = []
    print(solutions)
    for x in solutions:
        fitness_values.append(fitness_function(x, grid, landing_spot))
    # fitness_values = [sphere(x) for x in solutions]  # Evaluate the fitness of each solution
    es.tell(solutions, fitness_values)  # Update the CMA-ES parameters based on the fitness

# Get the best solution found by CMA-ES
best_solution = es.result.xbest
best_fitness = es.result.fbest

print("Best solution:", best_solution)
print("Best fitness value:", best_fitness)
