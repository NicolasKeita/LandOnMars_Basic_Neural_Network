from torch import nn
import torch.nn.functional as F


# TODO rename feed forward NN
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.hidden_layer = nn.Linear(n_observations, 512)  # Increase to 256 neurons
        self.output_layer = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return self.output_layer(x)
