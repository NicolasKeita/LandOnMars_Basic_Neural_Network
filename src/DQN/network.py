from torch import nn
import torch.nn.functional as F


# TODO rename feed forward NN

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.hidden_layer1 = nn.Linear(n_observations, 256)  # Change to 256 neurons
        self.hidden_layer2 = nn.Linear(256, 128)  # Add a second hidden layer with 128 neurons
        self.hidden_layer3 = nn.Linear(128, 64)  # Add a third hidden layer with 64 neurons
        self.output_layer = nn.Linear(64, n_actions)  # Adjust input size for the third hidden layer

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))  # Apply ReLU activation for the second hidden layer
        x = F.relu(self.hidden_layer3(x))  # Apply ReLU activation for the third hidden layer
        return self.output_layer(x)

# class DQN(nn.Module):
#     def __init__(self, n_observations, n_actions):
#         super(DQN, self).__init__()
#         self.hidden_layer = nn.Linear(n_observations, 512)  # Increase to 256 neurons
#         self.output_layer = nn.Linear(512, n_actions)
#
#     def forward(self, x):
#         x = F.relu(self.hidden_layer(x))
#         return self.output_layer(x)
