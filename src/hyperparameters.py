BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network


# mpc_pytorch
TIMESTEPS = 10  # T
N_BATCH = 1
LQR_ITER = 5
ACTION_LOW = 0.0 # TODO change to int
ACTION_HIGH = 4.0
ACTION_2_LOW = -90
ACTION_2_HIGH = 90

# mars
GRAVITY = 3.711
