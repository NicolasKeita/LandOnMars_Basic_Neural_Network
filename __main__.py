#!/usr/bin/env python
from src.create_environment import create_env
from src.learn_weights import learn_weights
import numpy as np


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
    return concatenated_numbers.reshape(num_lines, 2)


#
# RL is
#  - Optimization
#  - Delayed consequences
#  - Exploration
#  - Generalization (my agent has been trained with a specific environment,
#       but I'd like it to be effective on future unknown environment as well
#
if __name__ == '__main__':
    planet_surface = parse_planet_surface()
    env: list[list[bool]] = create_env(planet_surface, 7000, 3000)

    weights = learn_weights(planet_surface, None, env)
    print("----------- Learn Weight ends success")
