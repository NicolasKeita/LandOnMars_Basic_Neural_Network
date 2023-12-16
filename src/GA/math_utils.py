import math
from shapely import Point, LineString


def distance(a: Point, b: Point) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
    # return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def distance_2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    # return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distance_to_line(point, line_segments: LineString) -> float:
    point = Point(point)
    projected_point = line_segments.interpolate(line_segments.project(point))
    return distance(projected_point, point)

# def distance_squared_to_line(point, line_segments):
#     x, y = point
#     min_distance_squared = float('inf')
#
#     for segment in line_segments:
#         (x1, y1), (x2, y2) = segment
#         dx, dy = x2 - x1, y2 - y1
#
#         # Calculate the squared distance
#         dot_product = (x - x1) * dx + (y - y1) * dy
#         t = max(0, min(1, dot_product / (dx ** 2 + dy ** 2)))
#         closest_point = (x1 + t * dx, y1 + t * dy)
#
#         segment_distance_squared = distance_squared((x, y), closest_point)
#         min_distance_squared = min(min_distance_squared, segment_distance_squared)
#
#     return min_distance_squared
