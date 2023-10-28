import random
from pprint import pprint
from typing import List

from src.Action import Action
from src.create_environment import create_env


def create_legal_actions(previous_rotation: int | None = None, previous_power: int | None = None) -> List[Action]:
    return [Action(power, rotation)
            for rotation in range(-90, 90)
            for power in range(4)
            if (previous_rotation is None or abs(previous_rotation - rotation) <= 15)
            and (previous_power is None or abs(previous_power - power) <= 1)]


def create_q_table(mars_surface):
    env = create_env(mars_surface, 7000, 3000)
    env = [['S', 'F', 'F', 'F'],
           ['F', 'H', 'F', 'H'],
           ['F', 'F', 'F', 'H'],
           ['H', 'F', 'F', 'G']]
    states = [(y, x) for y in range(len(env)) for x in
              range(len(env[0]))]  # TODO include acceleration/hs/vs for it to be a Markrov system
    legal_actions = create_legal_actions()
    legal_actions = [0, 1, 2, 3]
    q_table = {}
    for state in states:
        for action in legal_actions:
            q_table[(state, action)] = 0
    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = pow(10, 4)

    for episode in range(num_episodes):
        state = (0, 0)
        done = False
        while not done:
            # Choose an action using epsilon-greedy policy
            if random.uniform(0, 1) < 0.2:
                action = random.choice(legal_actions)  # Explore
            else:
                action = max(legal_actions, key=lambda action: q_table[(state, action)])  # Exploit

            row, col = state
            new_row, new_col = row, col

            if action == 0:  # Left
                new_col = max(col - 1, 0)
            elif action == 1:  # Down
                new_row = min(row + 1, len(env) - 1)
            elif action == 2:  # Right
                new_col = min(col + 1, len(env[0]) - 1)
            else:  # Up
                new_row = max(row - 1, 0)

            new_state = (new_row, new_col)  # todo new state is equal to computer next turn rocket

            # Define rewards based on the environment
            reward = None
            if env[new_row][new_col] == 'H':  # TODO reward = -1 if not in the right area when landing.
                reward = -1  # Negative reward for falling into a hole
            elif env[new_row][new_col] == 'G':
                reward = 1  # Positive reward for reaching the goal
                done = True
            else:
                reward = 0  # No reward for other states

            # Update the Q-value for the current state-action pair
            best_next_action = max(legal_actions, key=lambda a: q_table[(new_state, a)])
            q_table[(state, action)] = \
                q_table[(state, action)] + \
                learning_rate * (reward + discount_factor * q_table[(new_state, best_next_action)] - q_table[
                    (state, action)])

            state = new_state
    pprint(q_table)
    return q_table
