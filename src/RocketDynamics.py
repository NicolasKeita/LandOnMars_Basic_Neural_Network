import math
import torch
from src.hyperparameters import GRAVITY






class RocketDynamics(torch.nn.Module):
    @staticmethod
    def forward(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        radians = action[1] * (math.pi / 180)
        x_acceleration = math.sin(radians) * action[0]
        y_acceleration = math.cos(radians) * action[0] - GRAVITY
        new_horizontal_speed = state[2] - x_acceleration
        new_vertical_speed = state[3] + y_acceleration
        new_x = state[0] + new_horizontal_speed - x_acceleration * 0.5
        new_y = state[1] + new_vertical_speed + y_acceleration * 0.5 + GRAVITY
        remaining_fuel = state[4] - action[0]
        new_state = new_x, new_y, new_horizontal_speed, new_vertical_speed, remaining_fuel, action[1], action[0]
        # TODO add the entire map of Mars to state
        return torch.tensor(new_state)
        # th = state[:, 0].view(-1, 1)
        # thdot = state[:, 1].view(-1, 1)
        #
        # g = 10
        # m = 1
        # l = 1
        # dt = 0.05
        #
        # u = action
        # u = torch.clamp(u, -2, 2)
        #
        # newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        # newth = th + newthdot * dt
        # newthdot = torch.clamp(newthdot, -8, 8)
        #
        # state = torch.cat((angle_normalize(newth), newthdot), dim=1)
        # return state


# def angle_normalize(x):
#     return (((x + math.pi) % (2 * math.pi)) - math.pi)