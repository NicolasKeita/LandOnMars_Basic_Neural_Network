class Point2D:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Point2D({self.x}, {self.y})"

    def __repr__(self) -> str:
        return f"Point2D({self.x}, {self.y})"

    def distance_to(self, other_point: 'Point2D') -> float:
        return ((self.x - other_point.x) ** 2 + (self.y - other_point.y) ** 2) ** 0.5
