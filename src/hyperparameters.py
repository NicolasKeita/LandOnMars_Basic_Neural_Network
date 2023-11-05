BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

# mpc_pytorch
TIMESTEPS = 10  # T
N_BATCH = 1
LQR_ITER = 5
ACTION_LOW = 0.0  # TODO change to int
ACTION_HIGH = 4.0
ACTION_2_LOW = -90
ACTION_2_HIGH = 90

# mars
GRAVITY = 3.711

# TODO create class GeneticAlgorithm and give this during init
# Genetic Algorithm Parameters
population_size = 10
generations_count = 100
mutation_rate = 0.1
dna_size = 100  # finite_horizon TODO (mars lander is 800 max CHANGE IT)

# Define the range of x values to search within
ACTION_1_INIT_RANGE = (-90, 90)
ACTION_2_INIT_RANGE = (0, 4)


def action_2_min_max(old_rota: int) -> tuple[int, int]:
    return max(old_rota - 15, -90), min(old_rota + 15, 90)


def action_1_min_max(old_power_thrust: int) -> tuple[int, int]:
    return max(old_power_thrust - 1, 0), min(old_power_thrust + 1, 4)
