import random
import time
from pprint import pprint
from typing import List

from src.Action import Action
from src.Rocket import State
from src.create_environment import create_env

# TODO move this info somewhere else:
# RL setting chosen : MDP (Markov Decision Process). #TODO revisit choices
# because
# I have access to a model (able to simulate next state),
# a reward function (able to know immediate reward) but
# I don't have access to a Value Function (no quick way to know the expected reward sum)
# and there are (finite) actions the agent is able take.
#
# then I am exploring MPC algorithm (Model Predictive Control) ... Not rdy yet.

# Model to -> Policy = Planning

# Policy-Based RL

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


def create_q_table(mars_surface):
    x_max = 7000
    y_max = 3000
    env: list[list[bool]] = create_env(mars_surface, x_max, y_max)

    for x, row in enumerate(env):
        for y, cell in enumerate(row):
            pass
    exit(0)


    num_episodes = 7000 * 3000

    for i in range(num_episodes):
        start_time = time.time()  # Record the start time
        print(i)

        # states = [[None] * y_max for _ in range(x_max)]
        array_2d = [(x, y) for y in range(y_max) for x in range(x_max)]
        # array_2d = [None for y in range(y_max) for x in range(x_max)]
        # states = [
        #     State(x=x, y=y)
        #     for y in range(y_max)
        #     for x in range(x_max)
        # ]

        end_time = time.time()  # Record the end time
        elapsed_time_ms = (end_time - start_time) * 1000  # Calculate the execution time in milliseconds
        elapsed_time_s = end_time - start_time  # Calculate the execution time in seconds

        print("EPISODE " + str(i) + " FINISHED")
        print(f"Episode {i} took {elapsed_time_ms:.2f} milliseconds ({elapsed_time_s:.2f} seconds)")
    exit(0)

    legal_actions = create_legal_actions()
    q_table = {}
    for state in states:
        for action in legal_actions:
            q_table[(state, action)] = 0
    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = pow(10, 4)

    w_vector = []

    for episode in range(num_episodes):
        state = (0, 0)
        done = False
        for action in legal_actions:
            while not done:
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
