from shapely import Point, LineString


def distance_squared(a: Point, b: Point) -> float:
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2


def distance_squared_to_closest_point_to_line_segments(point, line_segments):
    point = Point(point)
    line = LineString(line_segments)
    projected_point = line.interpolate(line.project(point))
    return distance_squared(projected_point, point)