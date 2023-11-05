import random

from src.hyperparameters import dna_size


# # action_ranges is a list of tuples, each containing the min and max values for an action.
def initialize_population(population_size: int, action_ranges: list[tuple[int | float, int | float]]) -> list[list[tuple]]:
    population = []
    for _ in range(population_size):
        dna = [
            tuple([
                random.randint(min_val, max_val) if isinstance(min_val, int) and isinstance(max_val, int)
                else random.uniform(min_val, max_val)
                for min_val, max_val in action_ranges
            ])
            for _ in range(dna_size)
        ]
        population.append(dna)
    return population


def evaluate_population(population, fitness_function, initial_state):
    return [fitness_function(initial_state, dna) for dna in population]
