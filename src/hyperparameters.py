# mars
GRAVITY = 3.711

# TODO create class GeneticAlgorithm and give this during init
# Genetic Algorithm Parameters
population_size = 100
generations_count = 2000
mutation_rate = 0.8
dna_size = 100  # finite_horizon TODO (mars lander is 800 max CHANGE IT)

# Define the range of x values to search within
ACTION_2_INIT_RANGE = (-90, 90)
ACTION_1_INIT_RANGE = (0, 4)


def action_2_min_max(old_rota: int) -> tuple[int, int]:
    return max(old_rota - 15, -90), min(old_rota + 15, 90)


def action_1_min_max(old_power_thrust: int) -> tuple[int, int]:
    return max(old_power_thrust - 1, 0), min(old_power_thrust + 1, 4)


def actions_min_max(action: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return action_1_min_max(action[0]), action_2_min_max(action[1])


def limit_actions(old_rota: int, old_power_thrust: int, action: tuple[int, int]):
    range_rotation = action_2_min_max(old_rota)
    rot = action[1] if range_rotation[0] <= action[1] <= range_rotation[1] else min(
        max(action[1], range_rotation[0]), range_rotation[1])
    range_thrust = action_1_min_max(old_power_thrust)
    thrust = action[0] if range_thrust[0] <= action[0] <= range_thrust[1] else min(
        max(action[0], range_thrust[0]), range_thrust[1])
    return rot, thrust
