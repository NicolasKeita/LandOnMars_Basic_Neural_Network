import random
from typing import Callable

from src.hyperparameters import dna_size, action_1_min_max, action_2_min_max, population_size
from src.mars_landing import fitness_function


# # action_ranges is a list of tuples, each containing the min and max values for an action.
def initialize_population(population_size: int,
                          action_ranges: list[tuple[int | float, int | float]]) -> list[list[tuple[int | float, ...]]]:
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
        population.append(dna)  # TODO Check if I can change dna to a tuple
    return population


def evaluate_population(population: list[list[tuple[int | float, ...]]],
                        fitness_function: Callable[[tuple, list[tuple], int], int],
                        initial_state: tuple, generation_id: int) -> list[int]:
    return [fitness_function(initial_state, dna, generation_id) for dna in population]


def select_population(population: list[list[tuple[int | float, ...]]], fitness_scores: list[int]) -> list[list[tuple[int | float, ...]]]:
    n_elites = 2
    sorted_population = [chromosome for score, chromosome in sorted(zip(fitness_scores, population), reverse=True)]
    return sorted_population[:n_elites]


# TODO introduce some Classes or check create custom type for type hinting only. current list[list[list[Types makes no senses
def crossover_population(population_survivors: list[list[tuple[int | float, ...]]]):
    offspring = []

    while len(offspring) < population_size - len(population_survivors):
        parent1 = random.choice(population_survivors)
        parent2 = random.choice(population_survivors)
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    offspring.extend(population_survivors)
    return offspring


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
