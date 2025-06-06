import torch.nn as nn

from .MSE import MSE
from .SSIM import SSIM


class MSE_SSIM(nn.Module):
    def __init__(
        self,
        data_range: float = 1.0,
        weights=[1, 1],
    ):
        super().__init__()

        self.data_range = data_range
        self.weights = weights

    def forward(self, prediction, target, mask=None):
        mse = MSE()(prediction, target, mask)
        ssim = SSIM(self.data_range).forward(prediction, target)

        return self.weights[0] * mse + self.weights[1] * ssim
