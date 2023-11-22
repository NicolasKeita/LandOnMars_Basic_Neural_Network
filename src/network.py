import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super(FeedForwardNN, self).__init__()
        self.device = device
        self.layer_1 = nn.Linear(in_dim, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, out_dim)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float).to(self.device)
        activation_1 = F.relu(self.layer_1(state))
        activation_2 = F.relu(self.layer_2(activation_1))
        activation_3 = F.relu(self.layer_3(activation_2))
        activation_4 = F.relu(self.layer_4(activation_3))
        raw_output = self.layer_5(activation_4)
        return raw_output