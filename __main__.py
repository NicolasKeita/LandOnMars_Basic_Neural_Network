#!/usr/bin/env python
from src.Point2D import Point2D
import math
from src.Rocket import Rocket
from src.create_environment import create_env
from src.learn_weights import learn_weights


def parse_mars_surface() -> list[Point2D]:
    return [Point2D(int(x), int(y)) for x, y in (input().split(' ') for _ in range(int(input())))]

#
# RL is
#  - Optimization
#  - Delayed consequences
#  - Exploration
#  - Generalization (my agent has been trained with a specific environment
#       but I'd like it to be effective on future unknown environment as well
#
if __name__ == '__main__':
    turn = 0
    mars_surface = parse_mars_surface()
    x_max = 7000
    y_max = 3000
    env: list[list[bool]] = create_env(mars_surface, x_max, y_max)
    init_rocket = Rocket(2500, 2700, 0, 0, 550, 0, 0, env)
    rocket = init_rocket
    print('INFO: this program is meant to be launched with an test-case as input.')

    weights = learn_weights(mars_surface, init_rocket, env)
    print("----------- Learn Weight ends success")
    exit(0)
    while True:
        turn += 1
        x, y, hs, vs = compute_next_turn_rocket(rocket)
        #legal_actions = create_legal_actions(rocket.rotation, rocket.power)  # TODO unused
        rocket = Rocket(x, y, hs, vs, rocket.state.fuel, rocket.state.rotation, rocket.state.power)
        time.sleep(0.1)
        if rocket.y <= 0:
            break
        rotation_chosen = rocket.rotation
        power_chosen = rocket.power
        print(
            f"Turn: {turn} nextX: {rocket.x} nextY: {rocket.y} My actions. Rocket Rotation: {rocket.rotation} Rocket Power: {rocket.power}")
        scatter.set_offsets([rocket.x, rocket.y])
        plt.pause(0.01)

    print(mars_surface, file=sys.stderr)
