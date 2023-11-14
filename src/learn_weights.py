from src.DQNAgent import DQNAgent
# from src.genetic_ann import *
# from src.GeneticAlgorithm import initialize_population, evaluate_population, select_population, mutate_population, \
#     crossover_population_1_k_point, uniform_crossover_population
from src.Point2D import Point2D
from src.Rocket import Rocket
from src.create_environment import RocketLandingEnv, create_env
from src.graph_handler import create_graph
# from src.create_environment import create_env, RocketLandingEnv
# from src.graph_handler import create_graph
# from src.hyperparameters import population_size, generations_count, mutation_rate
from src.linearQ import LinearQAgent
# from src.mars_landing import fitness_function
# import pandas as pd

# TODO move this info somewhere else:
# RL setting chosen : MDP (Markov Decision Process). #TODO revisit choices
# because
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
#   Policy Gradient vs Metaheuristic?
#       Ant Colony Optimization algorithm (model-based search)

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

# Classical control algorithm (PID / LQR / MPC) vs RL control algorithms
# bias = constant features

# Policy-Based RL
# The world outside my agent is stationary (independent of the agent actions).

def find_landing_spot(mars_surface: list[Point2D]) -> tuple[Point2D, Point2D]:
    for i in range(len(mars_surface) - 1):
        if mars_surface[i + 1].y == mars_surface[i].y:
            return mars_surface[i], mars_surface[i + 1]
    raise Exception('no landing site on test-case data')

# # store all active ANNs
# networks = []
# pool = []
# # Generation counter
# generation = 0
# # Initial Population
# population = 10
# for i in range(population):
#     networks.append(ANN())
# # Track Max Fitness
# max_fitness = 0
# # Store Max Fitness Weights
# optimal_weights = []

data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [10, 20, 30, 40, 50],
    'Feature3': [100, 200, 300, 400, 500],
    # ... add more features as needed
    'Label': [0, 1, 0, 1, 0]
}



def learn_weights(mars_surface: list[Point2D], init_rocket: Rocket, env):
    x_max = 7000
    y_max = 3000
    grid: list[list[bool]] = create_env(mars_surface, x_max, y_max)
    landing_spot = find_landing_spot(mars_surface)
    # initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
    initial_state = (2500, 2700, 0, 0, 550, 0, 0)
    create_graph(mars_surface, 'Landing on Mars')
    # env = gym.make('CartPole-v1')

    # Create and train the Linear Q-learning agent
    env = RocketLandingEnv(initial_state, landing_spot, grid)
    agent = DQNAgent(env.feature_amount, env.action_space_n, env)
    # agent = LinearQAgent(env)
    # agent.train(num_episodes=200)
    agent.pre_train()
    exit(97)
    state = env.reset()
    arr = []
    while True:
        action_index = agent.choose_action(state)
        arr.append(env.action_space[action_index])
        next_state, reward, done = agent.env.step(action_index)
        state = next_state
        if done:
            break
    print(arr)
    exit(0)

    # population = [create_neural_network() for _ in range(population_size)]
    population = initialize_population(population_size, [(0, 4), (-90, 90)])
    # networks = []
    # for i in range(population_size):
    #     networks.append(ANN())
    fitness_scores: list[float] = []
    for generation in range(generations_count):
        # for ann in networks:
        #     pass
            # Propagate to calculate fitness score
            # ann.forward_propagation(train_feature, train_label)
            # Add to pool after calculating fitness
            # pool.append(ann)

        # TODO RECALL put evaluate in a class
        fitness_scores = evaluate_population(population, train_data, train_labels, val_data, val_labels, initial_state, generation)
        selected_population = select_population(population, fitness_scores)
        # population = crossover_population_1_k_point(selected_population)
        if generation == generations_count - 1:  # TODO -1 right?
            break
        population = uniform_crossover_population(selected_population)
        population = mutate_population(population, mutation_rate)

        population.extend(selected_population)
        # print('gen count', generation)

    fitness_scores = evaluate_population(population, fitness_function, initial_state)
    best_chromosome = population[fitness_scores.index(max(fitness_scores))]
    print(best_chromosome)
    print("score : ", max(fitness_scores))
