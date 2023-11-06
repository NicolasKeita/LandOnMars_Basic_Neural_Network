import math


class Point2D:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Point2D({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Point2D({self.x}, {self.y})"

    @staticmethod
    def to_tuples(list_point2d: list['Point2D']) -> list[tuple[float, float]]:
        return [(obj.x, obj.y) for obj in list_point2d]

    def distance_to(self, other_point: 'Point2D') -> float:
        return math.sqrt((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2)
