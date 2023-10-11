import sys
import matplotlib.pyplot as plt
from typing import List
from src.Point2D import Point2D


def parse_mars_surface() -> List[Point2D]:
    return [Point2D(int(x), int(y)) for x, y in (input().split(' ') for _ in range(int(input())))]


def display_points_graph(points: List[Point2D]):
    x_coords, y_coords = zip(*[(point.x, point.y) for point in points])

    plt.plot(x_coords, y_coords, marker='o')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, 7000)
    plt.ylim(0, 3000)
    plt.title('Landing on Mars')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    mars = parse_mars_surface()
    display_points_graph(mars)
    print(mars, file=sys.stderr)
