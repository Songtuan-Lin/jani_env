import torch
import torch.nn as nn


class Policy(nn.Module):
    """A simple feedforward policy network."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [64, 64]):
        super(Policy, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)