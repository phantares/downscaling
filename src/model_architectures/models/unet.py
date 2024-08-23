import torch
import torch.nn as nn
import math


class UNet(nn.Module):
    def __init__(
        self, in_channels=1, out_channels=1, features=64, num_layers=3, **configs
    ):
        super().__init__()
        self.num_layers = num_layers

        self.input = nn.BatchNorm2d(in_channels)

        self.encoder = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.encoder.append(
                    ConvolutionBlock(in_channels, features, **configs["conv"])
                )
            else:
                self.encoder.append(
                    ConvolutionBlock(
                        features * 2 ** (layer - 1),
                        features * 2 ** (layer),
                        **configs["conv"],
                    )
                )

        self.maxpool = nn.ModuleList(
            [nn.MaxPool2d(**configs["sampling"]) for _ in range(num_layers - 1)]
        )

        self.deconv = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    features * 2 ** (layer),
                    features * 2 ** (layer - 1),
                    **configs["sampling"],
                )
                for layer in range(num_layers - 1, 0, -1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                ConvolutionBlock(
                    features * 2 ** (layer),
                    features * 2 ** (layer - 1),
                    **configs["conv"],
                )
                for layer in range(num_layers - 1, 0, -1)
            ]
        )

        self.output = nn.ModuleList(
            [
                *[
                    OutputConvolutionBlock(
                        features // 4 ** (i),
                        features // 4 ** (i + 1),
                        **configs["conv"],
                    )
                    for i in range(int(math.log(features, 4)) - int(math.log(4, 4)))
                ],
                nn.Conv2d(4, out_channels, **configs["conv"]),
                nn.ReLU(),
            ]
        )

    def forward(self, x):
        x = self.input(x)

        skip_connections = []
        for layer in range(self.num_layers):
            x = self.encoder[layer](x)

            if layer < self.num_layers - 1:
                skip_connections.insert(0, x)
                x = self.maxpool[layer](x)

        for layer in range(self.num_layers - 1):
            x = self.deconv[layer](x)
            x = torch.cat([x, skip_connections[layer]], dim=1)
            x = self.decoder[layer](x)

        for layer in self.output:
            x = layer(x)

        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **configs):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **configs),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, **configs),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class OutputConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **configs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **configs)
        self.relu = nn.ReLU(inplace=True)
        self.bnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bnorm(x)

        return x
