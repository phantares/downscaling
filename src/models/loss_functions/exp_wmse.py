import torch
import torch.nn as nn


class ExpWMSE(nn.Module):
    def __init__(self, b=5, c=1, scale=200):
        super().__init__()

        self.b = b
        self.c = c
        self.scale = scale
        self.register_buffer("max_weight", torch.exp(torch.tensor(b)))

    def forward(self, prediction, target):
        weights = torch.exp(self.b * ((target / self.scale) ** self.c))
        weights = torch.clamp(weights, max=self.max_weight)

        return torch.mean(weights * (prediction - target) ** 2)
