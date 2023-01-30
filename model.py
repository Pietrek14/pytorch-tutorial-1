import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # The model has only two nodes (one in the input layer, one in the output layer)
        # because it's supposed to predict a linear formula, which can be represented
        # with this model
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
