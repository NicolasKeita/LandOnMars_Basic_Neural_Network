import numpy as np


def distance_2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def distance_to_line(point, line_segments):
    x, y = point
    min_distance_squared = float('inf')

    for segment in line_segments:
        (x1, y1), (x2, y2) = segment.tolist()
        dx, dy = x2 - x1, y2 - y1

        dot_product = (x - x1) * dx + (y - y1) * dy
        t = max(0, min(1, dot_product / (dx ** 2 + dy ** 2)))
        closest_point = (x1 + t * dx, y1 + t * dy)

        segment_distance_squared = distance_2((x, y), closest_point)
        min_distance_squared = min(min_distance_squared, segment_distance_squared)

    return min_distance_squared

def calculate_intersection(x, y, new_pos, surface):
    def intersection_point(x1, y1, x2, y2, x3, y3, x4, y4):
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Lines are parallel or coincident

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point within both line segments
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return intersection_x, intersection_y

        return None  # Intersection point is outside one or both line segments

    # Check if the line segment intersects with the given surface
    for i in range(len(surface) - 1):
        intersect_point = intersection_point(x, y, *new_pos, *surface[i], *surface[i + 1])
        if intersect_point is not None:
            new_pos = np.array(intersect_point).flatten()
            break

    return new_pos
