import math
from shapely import Point, LineString, MultiPoint


def distance_squared(a: Point, b: Point) -> float:
    # return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def distance_squared_to_line(point, line_segments: LineString) -> float:
    point = Point(point)
    projected_point = line_segments.interpolate(line_segments.project(point))
    return distance_squared(projected_point, point)
