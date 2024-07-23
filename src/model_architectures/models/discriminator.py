import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=64, num_layers=5, **configs):
        super().__init__()

        self.num_layers = num_layers

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, features, **configs["conv"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, **configs["conv"]),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, **configs["conv"]),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, **configs["conv"]),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, features * 16, **configs["conv"]),
            nn.BatchNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 16, 1, **configs["output"]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        x = x.reshape(-1, 1)

        return x
