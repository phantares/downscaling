import torch
import torch.nn as nn

from .exp_wmse import ExpWMSE
from .ssim import SSIM


class ExpwSSIM(nn.Module):
    def __init__(
        self,
        b=3,
        c=3,
        scale_expw=200,
        scale_ssim=1,
        weights=[1, 1],
    ):
        super().__init__()

        self.expw = ExpWMSE(b, c, scale_expw)
        self.ssim = SSIM(scale_ssim)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, prediction, target, mask=None):
        expwmse = self.expw(prediction, target)
        ssim = self.ssim(prediction, target)

        return self.weights[0] * expwmse + self.weights[1] * ssim
