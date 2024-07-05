import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=64, num_layers=5, **configs):
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
            [nn.MaxPool2d(**configs["sampling"]) for _ in range(num_layers)]
        )

        self.output = nn.Sequential(
            nn.Conv2d(features * 2 ** (num_layers - 1), 1, kernel_size=3), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input(x)

        for layer in range(self.num_layers):
            x = self.encoder[layer](x)

            x = self.maxpool[layer](x)

        x = self.output(x)
        x = x.reshape(-1, 1)

        return x


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **configs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, **configs)
        self.conv2 = nn.Conv2d(out_channels, out_channels, **configs)
        self.relu = nn.LeakyReLU()
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bnorm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bnorm2(x)

        return x
