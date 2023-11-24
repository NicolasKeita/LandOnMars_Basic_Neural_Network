#!/usr/bin/env python
from src.create_environment import create_env, distance_to_line_segment, distance_to_surface, RocketLandingEnv
from src.graph_handler import create_graph
import numpy as np

from src.my_PPO import PPO


def parse_planet_surface():
    input_file = '''
        6
        0 100
        1000 500
        1500 100
        3000 100
        5000 1500
        6999 1000
    '''
    lines = input_file.strip().split('\n')
    num_lines = int(lines[0])
    concatenated_numbers = np.fromstring('\n'.join(lines[1:]), sep=' ')
    return concatenated_numbers.reshape(num_lines, 2).astype(int)


#
# RL is
#  - Optimization
#  - Delayed consequences
#  - Exploration
#  - Generalization (my agent has been trained with a specific environment,
#       but I'd like it to be effective on future unknown environment as well
#

# TODO move this info somewhere else:

# I have access to a model (given an action, I am able to predict what next state will be), I also am able to observe the next state but more importily I am able to predict,
# I also have a reward model, able to query a state and assign a reward to it without having to observe it. I can predict it.
# I can represent my model in different ways :
# table lookup model...
# TODO add more, stanford RL lecture 16
# a reward function (able to know immediate reward, which is 0 all the time expect when actually landing) but
# I don't have access to a Value Function (no quick way to know the expected reward sum)
# and there are (finite) actions the agent is able take.

# Despite having a model, I use model-free RL.
#
# ------------------            Metaheuristic        -------------------------#
# ---           Metaheuristic implicit        -       ------------------------#
# Seems like genetic algorithm is very good when there are multiple solutions (able quickly visit local-optimas)
# ---           Metaheuristic explicit        -       ------------------------#

# ------------------            Model-free RL        -------------------------#
# Usage of model-free because I can observe the next state after every action (the known model is allowing me to observe the next state after doing any action)
# Optimization - Policy Search
#   Policy Gradient methods vs Metaheuristic?
#       Particle Swarm Optimization algorithm
#       (special mention for Genetic algorithm,
#       generating hundreds of individuals each generation
#       felt too slow to converge)

# -------------------            Model-based RL       ------------------------#
# Only motive to use model-based RL would be that I have a model and
#   I cannot observe the states during the runs or the simulation or
#   The state space is small.
# Special mention for MCTS: that could work but,
#   I don't have the time (100ms max to compute next action) to even go over 1/100 of the tree and/or
#   I don't have the knowledge yet to generalize with weights that would fit into less than 50MB (CNN ?)

# Model-free RL is probably most suited for 90% of the problems. I guess that's why there is learning in RL.

# Planning = Compute a policy given a model - Value iteration / Policy iteration / dynamic programming - Policy Search / Q learning / dynamic programming - Approximate planning
# Planning algorithm :
# Value iteration
# Policy iteration
# Tree Search
# ...

# Control is a key aspect of this problem

# Genetic programming, linear regression, and decision trees up to
# a certain depth are examples of interpretable models, while neural networks
# and ensemble methods are considered black-boxes

# Classical control algorithm (PID / LQR / MPC) vs RL control algorithms
# bias = constant features

# Policy-Based RL
# The world outside my agent is stationary (independent of the agent actions).


def find_landing_spot(mars_surface):
    for i in range(len(mars_surface) - 1):
        if mars_surface[i + 1][1] == mars_surface[i][1]:
            landing_spot_start = (int(mars_surface[i][0]), int(mars_surface[i][1]))
            landing_spot_end = (int(mars_surface[i + 1][0]), int(mars_surface[i + 1][1]))
            return np.array([landing_spot_start, landing_spot_end])
    raise Exception('no landing site on test-case data')


def learn_weights(mars_surface: np.ndarray, init_rocket, env):
    x_max = 7000
    y_max = 3000
    grid: list[list[bool]] = create_env(mars_surface, x_max, y_max)
    landing_spot = find_landing_spot(mars_surface)
    landing_spot_points = []
    for x in range(landing_spot[0][0], landing_spot[1][0]):
        landing_spot_points.append(np.array([x, landing_spot[0][1]]))
    # initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
    # initial_state = (2500, 2700, 0, 0, 550, 0, 0)
    initial_state = np.array([
        2500, 2500, 0, 0, 500, 0, 0,
        distance_to_line_segment(np.array([500, 2700]), landing_spot_points),
        distance_to_surface(np.array([500, 2700]), mars_surface)
    ])
    # initial_state = np.concatenate([initial_state, mars_surface.flatten()])
    # create_graph(mars_surface, 'Landing on Mars')
    env = RocketLandingEnv(initial_state, landing_spot, grid, mars_surface, landing_spot_points)

    np.set_printoptions(suppress=True)
    # torch.autograd.set_detect_anomaly(True)

    my_proximal_policy_optimization = PPO(env)
    my_proximal_policy_optimization.learn(1000_000)



if __name__ == '__main__':
    planet_surface = parse_planet_surface()
    env: list[list[bool]] = create_env(planet_surface, 7000, 3000)

    weights = learn_weights(planet_surface, None, env)
    print("----------- Learn Weight ends success")
