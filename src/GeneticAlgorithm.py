import random
from typing import Callable

from matplotlib import cm, pyplot as plt
from matplotlib.axes import Axes

from src.hyperparameters import dna_size, action_1_min_max, action_2_min_max, population_size
from src.mars_landing import fitness_function


# # action_ranges is a list of tuples, each containing the min and max values for an action.
def initialize_population(population_size: int,
                          action_ranges: list[tuple[int | float, int | float]]) -> list[list[tuple[int | float, ...]]]:

    population = []
    for _ in range(population_size):
        dna = []
        # TODO future adapt to 3 or 4 actions,etc... and TODO if float random.uniform instead of randint
        previous_action_1 = 0  # TODO put this as param or something
        previous_action_2 = 0
        for _ in range(dna_size):
            action_1_min, action_1_max = action_1_min_max(previous_action_1)
            action_2_min, action_2_max = action_2_min_max(previous_action_2)
            gene = random.randint(action_1_min, action_1_max), random.randint(action_2_min, action_2_max)
            previous_action_1 = gene[0]
            previous_action_2 = gene[1]
            dna.append(gene)
        population.append(dna)  # TODO Check if I can change dna to a tuple
    return population


    # population = []
    # for _ in range(population_size):
    #     dna = [
    #         tuple([
    #             random.randint(min_val, max_val) if isinstance(min_val, int) and isinstance(max_val, int)
    #             else random.uniform(min_val, max_val)
    #             for min_val, max_val in action_ranges
    #         ])
    #         for _ in range(dna_size)
    #     ]
    #     population.append(dna)  # TODO Check if I can change dna to a tuple
    # return population


def evaluate_population(population: list[list[tuple[int | float, ...]]],
                        fitness_function: Callable[[tuple, list[tuple], int], int],
                        initial_state: tuple,
                        generation_id: int = 0) -> list[float]:
    result = []
    trajectories = []
    cmap = cm.get_cmap('Set1')
    color = cmap(generation_id)

    ax: Axes = plt.gca()

    # Clear previous trajectories
    for line in ax.lines:
        if line.get_label() != 'Mars Surface':
            line.remove()


    for dna in population:
        state_value, trajectory_x, trajectory_y = fitness_function(initial_state, dna, generation_id)
        result.append(state_value)
        trajectories.append((trajectory_x, trajectory_y))


    for trajectory in trajectories:
        plt.plot(trajectory[0], trajectory[1], marker='o', markersize=2, color=color, label=f'Rocket {generation_id}')
        plt.pause(0.001)
    return result


def select_population(population: list[list[tuple[int | float, ...]]], fitness_scores: list[int]) -> list[list[tuple[int | float, ...]]]:
    n_elites = 2
    tournament_size = 5  # Adjust the tournament size as needed

    selected_population = []

    while len(selected_population) < n_elites:
        # Randomly select individuals for the tournament
        tournament_individuals = random.sample(population, tournament_size)
        tournament_scores = [fitness_scores[population.index(individual)] for individual in tournament_individuals]

        # Select the winner of the tournament based on fitness scores
        winner_index = tournament_scores.index(max(tournament_scores))
        winner = tournament_individuals[winner_index]

        # Add the winner to the selected population
        selected_population.append(winner)
    return selected_population

# def select_population(population: list[list[tuple[int | float, ...]]], fitness_scores: list[float]) -> list[list[tuple[int | float, ...]]]:
#     n_elites = 2
#     sorted_population = [chromosome for score, chromosome in sorted(zip(fitness_scores, population), reverse=True)]
#     return sorted_population[:n_elites]


# TODO introduce some Classes or check create custom type for type hinting only. current list[list[list[Types makes no senses
def crossover_population_1_k_point(population_survivors: list[list[tuple[int | float, ...]]]):
    offspring = []

    while len(offspring) < population_size - len(population_survivors):
        parent1 = random.choice(population_survivors)
        parent2 = random.choice(population_survivors)
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.extend([child1, child2])
    return offspring


def uniform_crossover_population(population_survivors: list[list[tuple[int | float, ...]]]):
    offspring = []
    while len(offspring) < population_size - len(population_survivors):
        parent1 = random.choice(population_survivors)
        parent2 = random.choice(population_survivors)
        child: list[tuple[int, int]] = [(0, 0)] * len(parent1)
        for i in range(1, len(parent1)):
            child[i] = parent1[i] if random.random() > 0.50 else parent2[i]
        offspring.append(child)
    return offspring



def mutate_population(population_dna: list[list[tuple[int | float, ...]]], mutation_rate: float):
    mutated_population = []
    for dna in population_dna:
        if random.random() < mutation_rate:
            new_dna = []
            previous_action_1 = dna[0][0]
            previous_action_2 = dna[0][1]
            # print("dna", dna)
            for gene in dna:  # TODO future adapt to 3 or 4 actions,etc...
                # action_1_min_legal, action_1_max_legal = action_1_min_max(previous_action_1)
                # action_1_min_gene, action_1_max_gene = action_1_min_max(gene[0])
                # action_2_min_legal, action_2_max_legal = action_2_min_max(previous_action_2)
                # action_2_min_gene, action_2_max_gene = action_2_min_max(gene[1])
                # action_1_min = min(max(action_1_min_legal, action_1_min_gene), min(action_1_max_legal, action_1_max_gene))
                # action_1_max = max(min(action_1_max_legal, action_1_max_gene), max(action_1_min_legal, action_1_min_gene))
                # action_2_min = min(max(action_2_min_legal, action_2_min_gene), min(action_2_max_legal, action_2_max_gene))
                # action_2_max = max(min(action_2_max_legal, action_2_max_gene), max(action_2_min_legal, action_2_min_gene))
                # modified_gene = (random.randint(action_1_min, action_1_max),
                #                  random.randint(action_2_min, action_2_max))
                # modified_gene = (random.randint(action_1_min_gene, action_1_max_gene),
                #                  random.randint(action_2_min_gene, action_2_max_gene))
                modified_gene = (random.randint(0, 4), random.randint(-90, 90))
                new_dna.append(modified_gene)
                previous_action_1 = modified_gene[0]
                previous_action_2 = modified_gene[1]
            mutated_population.append(new_dna)
            # print("new_dna", new_dna)
        else:
            mutated_population.append(dna)
    return mutated_population
