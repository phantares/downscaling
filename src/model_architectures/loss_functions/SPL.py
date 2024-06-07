import torch
import torch.nn as nn
import itertools

from .weightedMSE import WeightedMSE
from .CRPS import CRPS


class SPL(nn.Module):

    def __init__(
        self,
        weights_MSE,
        thresholds_MSE,
        window_size: int = 9,
        integral_number: int = 1000,
        weights_SP=[1e-4, 1],
    ):
        super().__init__()

        self.weights_MSE = torch.tensor(weights_MSE).float()
        self.thresholds = thresholds_MSE
        self.window_size = window_size
        self.number = integral_number
        self.weights_SP = weights_SP

    def forward(self, prediction, target):
        prediction_disturbance = prediction - self._running_mean(
            prediction, self.window_size
        )
        target_disturbance = target - self._running_mean(target, self.window_size)

        mse = WeightedMSE(self.weights_MSE, self.thresholds).forward(prediction, target)
        crps = CRPS(self.number).forward(prediction_disturbance, target_disturbance)

        return self.weights_SP[0] * mse + self.weights_SP[1] * crps

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
