import cma
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

initial_mean = np.array([2500, 2700, 0, 0, 550, 0, 0])
sigma = 0.5  # Initial step size (standard deviation)
options = {'maxiter': 10000, 'tolfun': 1e-6}  # Maximum number of iterations and convergence criterion
es = cma.CMAEvolutionStrategy(initial_mean, sigma, options)

# Run the optimization loop
while not es.stop():
    solutions = es.ask()  # Sample new solutions
    fitness_values = []
    for x in solutions:
        fitness_values.append(fitness_function(x, grid, landing_spot, 550))
    es.tell(solutions, fitness_values)  # Update the CMA-ES parameters based on the fitness

# Get the best solution found by CMA-ES
best_solution = es.result.xbest
best_fitness = es.result.fbest

print("Best solution:", best_solution)
print("Best fitness value:", best_fitness)
