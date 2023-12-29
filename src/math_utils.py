import numpy as np


def distance_2(a, b) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def distance_to_line(x: float, y: float, line_segments: np.ndarray) -> float:
    x1, y1 = line_segments[:, 0, 0], line_segments[:, 0, 1]
    x2, y2 = line_segments[:, 1, 0], line_segments[:, 1, 1]
    dx, dy = x2 - x1, y2 - y1
    dot_product = (x - x1) * dx + (y - y1) * dy
    t = np.clip(dot_product / (dx ** 2 + dy ** 2), 0, 1)
    closest_point_x = x1 + t * dx
    closest_point_y = y1 + t * dy
    segment_distance_squared = (x - closest_point_x) ** 2 + (y - closest_point_y) ** 2
    min_distance_squared = np.min(segment_distance_squared)
    return min_distance_squared


def calculate_intersection(previous_pos: np.ndarray, new_pos: np.ndarray, surface: np.ndarray):
    x1, y1 = previous_pos
    x2, y2 = new_pos

    x3, y3 = surface[:-1, 0], surface[:-1, 1]
    x4, y4 = surface[1:, 0], surface[1:, 1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    mask = denominator != 0

    t = np.empty_like(denominator)
    u = np.empty_like(denominator)

    t[mask] = ((x1 - x3[mask]) * (y3[mask] - y4[mask]) - (y1 - y3[mask]) * (x3[mask] - x4[mask])) / denominator[mask]
    u[mask] = -((x1 - x2) * (y1 - y3[mask]) - (y1 - y2) * (x1 - x3[mask])) / denominator[mask]
    intersected_mask = (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)
    if np.any(intersected_mask):
        intersection_x = x1 + t[intersected_mask] * (x2 - x1)
        intersection_y = y1 + t[intersected_mask] * (y2 - y1)
        new_pos = np.array([intersection_x[0], intersection_y[0]])
    return new_pos


def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    return (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1]))


def do_segments_intersect(segment1: list[list[int]], segment2: list[list[int]]) -> bool:
    x1, y1 = segment1[0]
    x2, y2 = segment1[1]
    x3, y3 = segment2[0]
    x4, y4 = segment2[1]
    if (x1, y1) == (x3, y3) or (x1, y1) == (x4, y4) or (x2, y2) == (x3, y3) or (x2, y2) == (x4, y4):
        return False

    o1 = orientation((x1, y1), (x2, y2), (x3, y3))
    o2 = orientation((x1, y1), (x2, y2), (x4, y4))
    o3 = orientation((x3, y3), (x4, y4), (x1, y1))
    o4 = orientation((x3, y3), (x4, y4), (x2, y2))

    if (o1 != o2 and o3 != o4) or (o1 == 0 and on_segment((x1, y1), (x3, y3), (x2, y2))) or (
            o2 == 0 and on_segment((x1, y1), (x4, y4), (x2, y2))) or (
            o3 == 0 and on_segment((x3, y3), (x1, y1), (x4, y4))) or (
            o4 == 0 and on_segment((x3, y3), (x2, y2), (x4, y4))):
        if not on_segment((x1, y1), (x2, y2), (x3, y3)) and not on_segment((x1, y1), (x2, y2), (x4, y4)):
            return True
    return False
