from src.Point2D import Point2D


class Rocket:
    def __init__(self, x: float, y: float, hs: float, vs: float, fuel: float, rotation: int, power: int) -> None:
        self.x = x
        self.y = y
        self.hs = hs  # hs: the horizontal speed (in m/s), can be negative.
        self.vs = vs  # vs: the vertical speed (in m/s), can be negative.
        self.fuel = fuel  # fuel: the quantity of remaining fuel in liters.
        self.rotation = rotation  # rotation: the rotation angle in degrees (-90 to 90).
        self.power = power  # power: the thrust power (0 to 4).

    def get_pos(self):
        return Point2D(self.x, self.y)
