import random
from typing import Callable

from src.hyperparameters import dna_size, action_1_min_max, action_2_min_max
from src.mars_landing import fitness_function


# # action_ranges is a list of tuples, each containing the min and max values for an action.
def initialize_population(population_size: int, action_ranges: list[tuple[int | float, int | float]]) -> list[list[tuple[int | float, ...]]]:
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
        population.append(dna) # TODO Check if I can change dna to a tuple
    return population


def evaluate_population(population: list, fitness_function: Callable[[tuple, list[tuple[int, int]]], int],
                        initial_state: tuple) -> list[int]:
    return [fitness_function(initial_state, dna) for dna in population]


def select_population(population: list, fitness_scores: list[int]) -> list[list[tuple[int | float, ...]]]:
    return random.sample(population, 2)
    # selected_population = []
    # total_fitness = sum(fitness_scores)
    # for _ in range(len(population)):
    #     parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
    #     selected_population.append(parent1 if fitness_function(parent1) > fitness_function(parent2) else parent2)
    # return selected_population


def mutate_population(population_dna: list[list[tuple[int | float, ...]]], mutation_rate: float):
    mutated_population = []
    for dna in population_dna:
        if random.random() < mutation_rate:
            new_dna = []
            for gene in dna:  # TODO future adapt to 3 or 4 actions,etc...
                action_1_min, action_1_max = action_1_min_max(gene[0])
                action_2_min, action_2_max = action_2_min_max(gene[1])
                modified_gene = random.randint(action_1_min, action_1_max), random.randint(action_2_min, action_2_max)
                new_dna.append(modified_gene)
            mutated_population.append(new_dna)
        else:
            mutated_population.append(dna)
    return mutated_population
