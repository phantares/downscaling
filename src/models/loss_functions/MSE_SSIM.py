import torch.nn as nn

from .SSIM import SSIM


class MSE_SSIM(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        weights=[1, 1],
    ):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.weights = weights

    def forward(self, prediction, target):
        mse = nn.MSELoss()(prediction, target)
        ssim = SSIM(self.window_size, self.sigma, self.data_range).forward(
            prediction, target
        )

        return self.weights[0] * mse + self.weights[1] * ssim
