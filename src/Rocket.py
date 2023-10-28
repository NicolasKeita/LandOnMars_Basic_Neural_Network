from src.Point2D import Point2D


class RocketState:
    def __init__(self, x: float, y: float, hs: float, vs: float, fuel: float, rotation: int, power: int):
        # Features:
        self.x = x
        self.y = y
        self.hs = hs  # hs: the horizontal speed (in m/s), can be negative.
        self.vs = vs  # vs: the vertical speed (in m/s), can be negative.
        self.fuel = fuel  # the quantity of remaining fuel in liters.
        self.rotation = rotation  # the rotation angle in degrees (-90 to 90).
        self.power = power  # the thrust power (0 to 4).


class Rocket:
    def __init__(self, x: float, y: float, hs: float, vs: float, fuel: float, rotation: int, power: int):
        self.state = RocketState(x, y, hs, vs, fuel, rotation, power)

    def get_pos(self):
        return Point2D(self.state.x, self.state.y)
