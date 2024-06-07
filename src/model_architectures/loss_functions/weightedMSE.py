import torch
import torch.nn as nn


class WeightedMSE(nn.Module):

    def __init__(self, weights, thresholds):
        super().__init__()

        self.weights = torch.tensor(weights).float()
        self.thresholds = thresholds

    def forward(self, prediction, target):
        weights = self._calculate_weights_sum(target, self.weights, self.thresholds)

        return torch.mean(weights * (prediction - target) ** 2)

    def _calculate_weights_sum(self, target, weights, thresholds):
        weights_sum = (target >= thresholds[-1]).float() * weights[-1]

        for i, threshold in enumerate(thresholds[:-1]):
            weights_sum = (
                weights_sum
                + ((target >= threshold) & (target < thresholds[i + 1])).float()
                * weights[i]
            )

        return weights_sum
