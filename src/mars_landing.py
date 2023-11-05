#initial_state = (2500, 2700, 0, 0, 550, 0, 0, env, landing_spot)
def reward_function(state):
    rocket_pos_x = state[0]
    rocket_pos_y = state[1]
    remaining_fuel = state[4]
    env = state[7]
    landing_spot = state[8]
    if landing_spot[0].x <= rocket_pos_x <= landing_spot[1].x and landing_spot[0].y >= rocket_pos_y:  # TODO fix accuracy for reward
        return remaining_fuel
    if remaining_fuel < -4:
        return -1
    if not env[rocket_pos_x][rocket_pos_y]:  # TODO x and y may be inverted there
        return -1
    return 0


def compute_next_state(state, action):
    return state


# TODO namespace mars_landing
# AKA state-value function V_pi(s)
def fitness_function(state, dna: list[tuple[int, int]]) -> int:
    state_value = 0
    for gene in dna:
        state = compute_next_state(state, gene)
        state_value = reward_function(state)
        if state_value != 0:
            break
    return state_value
