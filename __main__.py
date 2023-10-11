import sys
import matplotlib.pyplot as plt
from typing import List
from src.Point2D import Point2D
import time


def parse_mars_surface() -> List[Point2D]:
    return [Point2D(int(x), int(y)) for x, y in (input().split(' ') for _ in range(int(input())))]


def display_points_graph(points: List[Point2D], isolated_point: Point2D, title: str):
    x_coords, y_coords = zip(*[(point.x, point.y) for point in points])

    plt.plot(x_coords, y_coords, marker='o', label='Mars Surface')
    plt.scatter(isolated_point.x, isolated_point.y, color='red', label='Rocket')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, 7000)
    plt.ylim(0, 3000)
    plt.title(title)
    # plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    turn = 0
    rocket = Point2D(2500, 2700)
    mars = parse_mars_surface()
    while True:
        turn += 1
        rocket.x -= 50
        rocket.y -= 50
        time.sleep(1)
        if rocket.y <= 0:
            break
        print(turn)
        display_points_graph(mars, rocket, 'Landing on Mars')
    print(mars, file=sys.stderr)
