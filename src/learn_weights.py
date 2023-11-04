import math
import random
import time
from pprint import pprint

from mpc import mpc
import torch

from src.Action import Action
from src.Point2D import Point2D
from src.Rocket import State, Weight, Rocket
from src.create_environment import create_env
from src.RocketDynamics import RocketDynamics
from src.hyperparameters import TIMESTEPS, ACTION_LOW, ACTION_HIGH, LQR_ITER, N_BATCH

# TODO move this info somewhere else:
# RL setting chosen : MDP (Markov Decision Process). #TODO revisit choices
# because
# I have access to a model (given an action, I am able to predict what next state will be),
# a reward function (able to know immediate reward, which is 0 all the time expect when actually landing) but
# I don't have access to a Value Function (no quick way to know the expected reward sum)
# and there are (finite) actions the agent is able take.
#
# then I am exploring MPC algorithm (Model Predictive Control) ... Not rdy yet.

# Model to -> Policy = Planning

# Policy-Based RL
# The world outside my agent is stationary (independent of the agent actions).

gravity = 3.711


# TODO use this, somewhere
def create_legal_actions(previous_rotation: int | None = None, previous_power: int | None = None) -> list[Action]:
    rotation_max = 90
    power_max = 4
    return [Action(power, rotation)
            for rotation in range(-rotation_max, rotation_max)
            for power in range(power_max)
            if (previous_rotation is None or abs(previous_rotation - rotation) <= 15)
            and (previous_power is None or abs(previous_power - power) <= 1)]


def control_process():
    # # Choose an action using epsilon-greedy policy
    # if random.uniform(0, 1) < 0.2:
    #     action = random.choice(legal_actions)  # Explore
    # else:
    #     action = max(legal_actions, key=lambda action: q_table[(state, action)])  # Exploit
    pass


# Define the policy (e.g., epsilon-greedy)
def extract_features(state: tuple[float, float, float, float, float, int, int], env: list[list[bool]]):
    concatenated_data = list(state) + [item for sublist in env for item in sublist]
    return tuple(concatenated_data)
    # features = list(state)
    # for x, row in enumerate(env):
    #     for y, value in enumerate(row):
    #         features.append((f"{x},{y}", value))
    # return features


def policy(state: tuple[float, float, float, float, float, int, int], epsilon, weights, env) -> tuple[int, int]:
    if random.random() < epsilon:
        # Exploration: Choose a random action
        return random.randint(0, 4), random.randint(-90, 90)
    else:
        # Exploitation: Choose the action with the highest estimated value
        features = extract_features(state, env)
        values = [sum(w * f for w, f in zip(weights, features)) for _ in range(len(features))]
        return (values.index(max(values)), 0) # TODO fix


num_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

def find_landing_spot(mars_surface: list[Point2D])-> list[tuple[Point2D, Point2D]]:
    seen_y_values = {}
    return [(seen_y_values.setdefault(point.y, point), point) for point in mars_surface if point.y in seen_y_values]


def reward_function(state, action, env, mars_surface):
    landing_spot = find_landing_spot(mars_surface)
    print(landing_spot)
    # if env[state[0]][state[1]]:

    # pass


def cost_function(state):
    return 7500 - state[4]


def take_action(state: tuple[float, float, float, float, float, int, int], action: tuple[int, int], env, mars_surface):
    radians = action[1] * (math.pi / 180)
    x_acceleration = math.sin(radians) * action[0]
    y_acceleration = math.cos(radians) * action[0] - gravity
    new_horizontal_speed = state[2] - x_acceleration
    new_vertical_speed = state[3] + y_acceleration
    new_x = state[0] + new_horizontal_speed - x_acceleration * 0.5
    new_y = state[1] + new_vertical_speed + y_acceleration * 0.5 + gravity
    remaining_fuel = state[4] - action[0]
    new_state = new_x, new_y, new_horizontal_speed, new_vertical_speed, remaining_fuel, action[1], action[0]
    reward = reward_function(state, action, env, mars_surface)
    return new_state, reward


def learn_weights(mars_surface: list[Point2D], init_rocket: Rocket, env):
    x_max = 7000
    y_max = 3000
    env: list[list[bool]] = create_env(mars_surface, x_max, y_max)
    feature_names = ['x', 'y', 'hs', 'vs', 'fuel', 'rotation', 'power']

    weights = tuple(0.0 for _ in range(720 + len(feature_names)))
    # weights = {feature_name: 0.0 for feature_name in feature_names}
    # weights.update({f"{x},{y}": 0.0 for x, row in enumerate(env) for y, _ in enumerate(row)})

    # cost_function = None
    cost_function = mpc.QuadCost(torch.tensor([0, 0, 0, 0, 1, 0, 0]), torch.tensor([0, 0, 0, 0, 1, 0, 0]))
    total_reward = 0
    u_init = None
    nx = 2
    nu = 1

    goal_weights = torch.tensor((1., 0.1))  # nx
    goal_state = torch.tensor((0., 0.))  # nx
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(nu)
    ))  # nx + nu
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(nu)))
    Q = torch.diag(q).repeat(TIMESTEPS, N_BATCH, 1, 1)  # T x B x nx+nu x nx+nu
    p = p.repeat(TIMESTEPS, N_BATCH, 1)
    cost_function = mpc.QuadCost(Q, p)  # T x B x nx+nu (linear component of cost)

    for episode in range(num_episodes):
        episode_states = []
        episode_rewards = []

        # state = (0, 0)
        state = init_rocket.state
        state = torch.tensor(state)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost_function, RocketDynamics())
        action = nominal_actions[0]  # take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        # s, r, _, _ = env.step(action.detach().numpy())
        # total_reward += r


        # while True:
        #     exit(0)
        #     action = policy(state, epsilon, weights, env)
        #     next_state, reward = take_action(state, action, env, mars_surface)  # Define take_action() based on the environment
        #     exit(0)
        #     episode_states.append(state)
        #     episode_rewards.append(reward)
        #     state = next_state
