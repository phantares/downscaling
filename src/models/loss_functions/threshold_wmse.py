import torch
import torch.nn as nn


class ThresholdWMSE(nn.Module):

    def __init__(self, weights, thresholds):
        super().__init__()

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer(
            "thresholds", torch.tensor(thresholds, dtype=torch.float32)
        )

    def forward(self, prediction, target):
        weights = self._calculate_weights_sum(target)
        return torch.mean(weights * (prediction - target) ** 2)

    def _calculate_weights_sum(self, target):
        bin_index = torch.bucketize(target, self.thresholds, right=True)
        return self.weights[bin_index]
