from src.Point2D import Point2D


class Rocket:
    def __init__(self, x: float, y: float, hs: float, vs: float, f: float, r: float, p: float) -> None:
        self.x = x
        self.y = y
        self.hs = hs
        self.vs = vs
        self.f = f
        self.r = r
        self.p = p

        # hs: the horizontal speed (in m/s), can be negative.
        # vs: the vertical speed (in m/s), can be negative.
        # f: the quantity of remaining fuel in liters.
        # r: the rotation angle in degrees (-90 to 90).
        # p: the thrust power (0 to 4).

    def get_pos(self):
        return Point2D(self.x, self.y)
