# Custom neural net created with the nn.Module. Used in older version of project

import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        self.linear_relu_stack = nn.Sequential(
            self.l1,
            self.relu,
            self.l2,
            self.relu,
            self.l3,
        )
    
    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


