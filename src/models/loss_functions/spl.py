import torch
import torch.nn as nn

from .threshold_wmse import ThresholdWMSE
from .crps import CRPS


class SPL(nn.Module):

    def __init__(
        self,
        weights_wmse,
        thresholds_wmse,
        window_size: int = 9,
        integral_number: int = 1000,
        weights=[1e-4, 1],
    ):
        super().__init__()

        self.wmse = ThresholdWMSE(weights_wmse, thresholds_wmse)
        self.crps = CRPS(window_size, integral_number)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, prediction, target):
        wmse = self.wmse(prediction, target)
        crps = self.crps(prediction, target)

        return self.weights[0] * wmse + self.weights[1] * crps
