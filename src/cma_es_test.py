import cma
import numpy as np

# Define the Sphere function (objective function to be minimized)
def sphere(x):
    return sum(x**2)

# Set up the CMA-ES optimizer
initial_mean = np.zeros(10)  # Initial guess for the mean vector
sigma = 0.5  # Initial step size (standard deviation)
options = {'maxiter': 1000, 'tolfun': 1e-6}  # Maximum number of iterations and convergence criterion
es = cma.CMAEvolutionStrategy(initial_mean, sigma, options)

# Run the optimization loop
while not es.stop():
    solutions = es.ask()  # Sample new solutions
    fitness_values = [sphere(x) for x in solutions]  # Evaluate the fitness of each solution
    es.tell(solutions, fitness_values)  # Update the CMA-ES parameters based on the fitness

# Get the best solution found by CMA-ES
best_solution = es.result.xbest
best_fitness = es.result.fbest

print("Best solution:", best_solution)
print("Best fitness value:", best_fitness)
