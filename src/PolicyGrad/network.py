import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer_1 = nn.Linear(in_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, out_dim)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
        activation_1 = F.relu(self.layer_1(state))
        activation_2 = F.relu(self.layer_2(activation_1))
        raw_output = self.layer_3(activation_2)
        return raw_output
        output_dim1 = torch.round(torch.tanh(raw_output[0]) * 90.0)
        output_dim2 = torch.round(torch.sigmoid(raw_output[1]) * 4.0)
        output = torch.tensor([output_dim1, output_dim2], dtype=torch.float)
        return output

class FeedForwardNN_critic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer_1 = nn.Linear(in_dim, 64)
        self.layer_2 = nn.Linear(64, 64)
        self.layer_3 = nn.Linear(64, out_dim)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)
        activation_1 = F.relu(self.layer_1(state))
        activation_2 = F.relu(self.layer_2(activation_1))
        raw_output = self.layer_3(activation_2)
        return output
