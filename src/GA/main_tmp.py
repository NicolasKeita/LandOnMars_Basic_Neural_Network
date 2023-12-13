import gymnasium as gym
import numpy as np

# OpenAI Gym environment
env = gym.make('CartPole-v1')

# Genetic Algorithm Parameters
population_size = 50
num_generations = 50
mutation_rate = 0.1

# Function to initialize the population
def initialize_population():
    return [np.random.rand(4) * 2 - 1 for _ in range(population_size)]

# Function to calculate the fitness of an individual
def calculate_fitness(individual):
    total_reward = 0
    for _ in range(5):  # Run the episode 5 times to get a more stable fitness
        state, _ = env.reset()
        for _ in range(200):  # Run for a maximum of 200 timesteps
            action = int(np.dot(individual, state) > 0)  # Simple threshold-based policy
            state, reward, done,_, _ = env.step(action)
            total_reward += reward
            if done:
                break
    return total_reward / 5  # Average over 5 runs

# Function for tournament selection
def tournament_selection(population, fitness_values, tournament_size=3):
    tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_index]

# Genetic Algorithm
population = initialize_population()

for generation in range(num_generations):
    # Calculate fitness for each individual in the population
    fitness_values = [calculate_fitness(ind) for ind in population]

    # Select parents using tournament selection
    parents = [tournament_selection(population, fitness_values) for _ in range(population_size)]

    # Crossover (single-point crossover)
    crossover_point = np.random.randint(4)
    offspring = [np.concatenate((parents[i][:crossover_point], parents[i+1][crossover_point:]))
                 for i in range(0, population_size, 2)]

    # Mutation
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            mutation_point = np.random.randint(4)
            offspring[i][mutation_point] += np.random.randn()

    # Replace the old population with the new population
    population = offspring

# Find the best individual in the final population
final_fitness_values = [calculate_fitness(ind) for ind in population]
best_individual = population[np.argmax(final_fitness_values)]

# Evaluate the best individual
total_reward_best = calculate_fitness(best_individual)

print(f"Best Individual: {best_individual}")
print(f"Total Reward with Best Individual: {total_reward_best}")

# Close the Gym environment
env.close()
