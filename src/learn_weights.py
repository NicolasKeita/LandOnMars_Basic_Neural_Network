import random
import time
from pprint import pprint
from typing import List

from src.Action import Action
from src.Point2D import Point2D
from src.Rocket import State, Weight, Rocket
from src.create_environment import create_env


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

def create_legal_actions(previous_rotation: int | None = None, previous_power: int | None = None) -> List[Action]:
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
def policy(state: State, epsilon, weights) -> tuple[int, int]:
    if random.random() < epsilon:
        # Exploration: Choose a random action
        return random.randint(0, 4), random.randint(-90, 90)
    else:
        # Exploitation: Choose the action with the highest estimated value
        #features = extract_features(state)
        features = state.features
        values = [sum(w * f for w, f in zip(weights, features)) for _ in range(len(features))]
        return values.index(max(values))


num_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1


def learn_weights(mars_surface: list[Point2D], init_rocket: Rocket):
    x_max = 7000
    y_max = 3000
    env: list[list[bool]] = create_env(mars_surface, x_max, y_max)
    feature_names = ['x', 'y', 'hs', 'vs', 'fuel', 'rotation', 'power']

    weights = {feature_name: 0.0 for feature_name in feature_names}
    weights.update({f"{x},{y}": 0.0 for x, row in enumerate(env) for y, _ in enumerate(row)})

    for episode in range(num_episodes):
        episode_states = []
        episode_rewards = []

        # state = (0, 0)
        state = init_rocket.state
        while True:
            action = policy(state, epsilon, weights)
            next_state, reward = take_action(state, action)  # Define take_action() based on the environment
            episode_states.append(state)
            episode_rewards.append(reward)
            state = next_state
