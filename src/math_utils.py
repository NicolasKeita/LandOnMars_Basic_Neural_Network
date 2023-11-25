from shapely import Point


def distance_squared(a: Point, b: Point) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
