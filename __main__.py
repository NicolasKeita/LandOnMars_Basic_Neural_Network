import sys
import matplotlib.pyplot as plt
from typing import List
from src.Point2D import Point2D
import time
import math
from src.Rocket import Rocket

gravity = 3.711


def parse_mars_surface() -> List[Point2D]:
    return [Point2D(int(x), int(y)) for x, y in (input().split(' ') for _ in range(int(input())))]


def create_graph(points: List[Point2D], rocket_position: Point2D, title: str):
    plt.xlim(0, 7000)
    plt.ylim(0, 3000)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.plot(*zip(*[(point.x, point.y) for point in points]), marker='o', label='Mars Surface')
    plt.legend()
    plt.grid(True)
    plt.draw()


def compute_next_turn_rocket(rocket: Rocket):
    radians = rocket.r * (math.pi / 180)
    x_acceleration = math.sin(radians) * rocket.p
    y_acceleration = math.cos(radians) * rocket.p - gravity
    new_horizontal_speed = rocket.hs - x_acceleration
    new_vertical_speed = rocket.vs + y_acceleration
    new_x = rocket.x + new_horizontal_speed - x_acceleration * 0.5
    new_y = rocket.y + new_vertical_speed + y_acceleration * 0.5 + gravity
    return new_x, new_y, new_horizontal_speed, new_vertical_speed


if __name__ == '__main__':
    turn = 0
    init_rocket = Rocket(2500, 2700, 0, 0, 550, 0, 0)
    mars = parse_mars_surface()

    rocket = init_rocket
    scatter = plt.scatter(rocket.x, rocket.y, color='red', label='Rocket')
    create_graph(mars, init_rocket.get_pos(), 'Landing on Mars')
    while True:
        turn += 1
        x, y, hs, vs = compute_next_turn_rocket(rocket)
        rocket = Rocket(x, y, hs, vs, rocket.f, rocket.r, rocket.p)
        time.sleep(0.1)
        if rocket.y <= 0:
            break
        print(turn, rocket.x, rocket.y)
        scatter.set_offsets([rocket.x, rocket.y])
        plt.pause(0.01)

    print(mars, file=sys.stderr)
