from enum import Enum
import torch.nn as nn

from .weightedMSE import WeightedMSE
from .CRPS import CRPS
from .SPL import SPL
from .SSIM import SSIM
from .MSE_SSIM import MSE_SSIM
from .expWMSE import ExpWMSE
from .MSE import MSE


class LossType(Enum):
    MSE = MSE
    weightedMSE = WeightedMSE
    CRPS = CRPS
    SPL = SPL
    SSIM = SSIM
    MSE_SSIM = MSE_SSIM
    expWMSE = ExpWMSE
