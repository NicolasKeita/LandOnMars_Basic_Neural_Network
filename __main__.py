#!/usr/bin/env python
from src.create_environment import create_env
from src.learn_weights import learn_weights
import numpy as np


def parse_planet_surface():
    input_file = '''
        20
        0 1000
        300 1500
        350 1400
        500 2000
        800 1800
        1000 2500
        1200 2100
        1500 2400
        2000 1000
        2200 500
        2500 100
        2900 800
        3000 500
        3200 1000
        3500 2000
        3800 800
        4000 200
        5000 200
        5500 1500
        6999 2800
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
if __name__ == '__main__':
    planet_surface = parse_planet_surface()
    env: list[list[bool]] = create_env(planet_surface, 7000, 3000)

    weights = learn_weights(planet_surface, None, env)
    print("----------- Learn Weight ends success")
