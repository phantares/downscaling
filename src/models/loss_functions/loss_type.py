from enum import Enum

from .weightedMSE import WeightedMSE
from .CRPS import CRPS
from .SPL import SPL
from .SSIM import SSIM
from .MSE_SSIM import MSE_SSIM


class LossType(Enum):
    weightedMSE = WeightedMSE
    CRPS = CRPS
    SPL = SPL
    SSIM = SSIM
    MSE_SSIM = MSE_SSIM
