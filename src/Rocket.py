from src.Point2D import Point2D


class Weight:
    def __init__(self, feature_name: str, weight: float):
        self.feature_name = feature_name
        self.weight = weight


class Feature:
    def __init__(self, name: str, value: any, weight: float | None):
        self.name = name
        self.value = value
        self.weight = weight  # TODO should I limit the weight to [-1, 1] (throwing error if not)?


# class State:
#     # def __init__(self, x: float | None, y: float | None, hs: float | None, vs: float | None, fuel: float | None, rotation: int | None, power: int | None):
#     def __init__(self, x: float | None = None, y: float | None = None, hs: float | None = None,
#                  vs: float | None = None, fuel: float | None = None, rotation: int | None = None,
#                  power: int | None = None):
#         self.features: list[Feature] = []
#         self.features.append(Feature('x', x, None))
#         self.features.append(Feature('y', y, None))
#         self.features.append(Feature('hs', hs, None))  # hs: the horizontal speed (in m/s), can be negative.
#         self.features.append(Feature('vs', vs, None))  # vs: the vertical speed (in m/s), can be negative.
#         self.features.append(Feature('fuel', fuel, None))  # the quantity of remaining fuel in liters.
#         self.features.append(Feature('rotation', rotation, None))  # the rotation angle in degrees (-90 to 90).
#         self.features.append(Feature('power', power, None))  # the thrust power (0 to 4).
#

# class State:
#     def __init__(self, x: float | None = None, y: float | None = None, hs: float | None = None,
#                  vs: float | None = None, fuel: float | None = None, rotation: int | None = None,
#                  power: int | None = None):
#         self.features: list[Feature] = [
#             Feature('x', x, None),
#             Feature('y', y, None),
#             Feature('hs', hs, None),
#             Feature('vs', vs, None),
#             Feature('fuel', fuel, None),
#             Feature('rotation', rotation, None),
#             Feature('power', power, None)
#         ]


class State:
    def __init__(self, **kwargs):
        self.features = [Feature(key, value, None) for key, value in kwargs.items()]


class Rocket:
    def __init__(self, x: float, y: float, hs: float, vs: float, fuel: float, rotation: int, power: int,
                 env: list[list[bool]]):
        self.state = State(x=x, y=y, hs=hs, vs=vs, fuel=fuel, rotation=rotation, power=power)
        for x, row in enumerate(env):
            for y, value in enumerate(row):
                self.state.features.append(Feature(f"{x},{y}", value, 0.0))

    def get_pos(self):
        x = (feature.value for feature in self.state.features if feature.name == 'x')
        y = (feature.value for feature in self.state.features if feature.name == 'y')
        return Point2D(float(x), float(y))
