import torch
import torch.nn as nn
import itertools

from .threshold_wmse import ThresholdWMSE
from .crps import CRPS


class SPL(nn.Module):

    def __init__(
        self,
        weights_MSE,
        thresholds_MSE,
        window_size: int = 9,
        integral_number: int = 1000,
        weights=[1e-4, 1],
    ):
        super().__init__()

        self.window_size = window_size

        self.wmse = ThresholdWMSE(weights_MSE, thresholds_MSE)
        self.crps = CRPS(integral_number)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, prediction, target):
        prediction_disturbance = prediction - self._running_mean(
            prediction, self.window_size
        )
        target_disturbance = target - self._running_mean(target, self.window_size)

        wmse = self.wmse.forward(prediction, target)
        crps = self.crps.forward(prediction_disturbance, target_disturbance)

        return self.weights[0] * wmse + self.weights[1] * crps

    def _running_mean(self, data, window_size: int, dim: int = 2):
        padding = window_size // 2

        data_shape = data.shape
        data_with_padding_shape = [dimension for dimension in data_shape]
        data_with_padding_shape[-dim:] = [
            dimension + padding * 2 for dimension in data_with_padding_shape[-dim:]
        ]

        data_for_running = torch.zeros(data_with_padding_shape, device=data.device)
        weight = torch.zeros(data_with_padding_shape, device=data.device)

        window = itertools.product(*[range(2 * padding + 1) for _ in range(dim)])
        for window_grid in window:

            weight_multiplier = 1
            if window_size % 2 == 0:
                for i in window_grid:
                    if i == 0 or i == window_size:
                        weight_multiplier *= 0.5

            data_index = [
                slice(start, start + data_shape[-(dim - i)], 1)
                for i, start in enumerate(window_grid)
            ]
            data_index = [Ellipsis, *data_index]

            data_for_running[tuple(data_index)] += data
            weight[tuple(data_index)] += (
                torch.ones(data_shape, device=data.device) * weight_multiplier
            )

        data_index[-dim:] = [
            slice(padding, data_with_padding_shape[-(dim - i)] - padding, 1)
            for i in range(dim)
        ]

        return data_for_running[tuple(data_index)] / weight[tuple(data_index)]
