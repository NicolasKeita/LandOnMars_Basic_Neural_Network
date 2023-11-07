# mars
GRAVITY = 3.711

# TODO create class GeneticAlgorithm and give this during init
# Genetic Algorithm Parameters
population_size = 8
generations_count = 1000
mutation_rate = 0.2
dna_size = 100  # finite_horizon TODO (mars lander is 800 max CHANGE IT)

# Define the range of x values to search within
ACTION_2_INIT_RANGE = (-90, 90)
ACTION_1_INIT_RANGE = (0, 4)


def action_2_min_max(old_rota: int) -> tuple[int, int]:
    return max(old_rota - 15, -90), min(old_rota + 15, 90)


def action_1_min_max(old_power_thrust: int) -> tuple[int, int]:
    return max(old_power_thrust - 1, 0), min(old_power_thrust + 1, 4)
