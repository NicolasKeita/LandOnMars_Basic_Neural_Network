#initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
def reward_function(state):
    if state[4] < -4:
        return -1
    if state[7][state[0]][state[1]] == True:
        return -1
    return 0


def compute_next_state(state, action):
    return state


# TODO namespace mars_landing
# AKA state-value function V_pi(s)
def fitness_function(state, dna: list[tuple[int, int]]) -> int:
    state_value = 0
    for gene in dna:
    # while state_value == 0:
        state = compute_next_state(state, gene)
        state_value = reward_function(state)
        break;
    return state_value