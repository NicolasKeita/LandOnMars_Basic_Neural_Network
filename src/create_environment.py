from typing import List
from src.Point2D import Point2D


def create_environment(surface_points: list[Point2D], x_max: int, y_max: int) -> List[List[bool]]:
    def surface_function(x, sorted_points):
        for i in range(len(sorted_points) - 1):
            x1, y1 = sorted_points[i].x, sorted_points[i].y
            x2, y2 = sorted_points[i + 1].x, sorted_points[i + 1].y
            if x1 <= x <= x2:
                return round(y1 + (x - x1) * (y2 - y1) / (x2 - x1))
        return 0

    world = [[False] * x_max for _ in range(y_max)]
    sorted_points = sorted(surface_points, key=lambda p: p.x)

    for x in range(x_max):
        for y in range(surface_function(x, sorted_points), y_max):
            world[y][x] = True
    return world
