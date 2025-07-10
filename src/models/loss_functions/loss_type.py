from enum import Enum

from .mse import MSE
from .threshold_wmse import ThresholdWMSE
from .crps import CRPS
from .spl import SPL
from .SSIM import SSIM
from .exp_wmse import ExpWMSE
from .expw_ssim import ExpwSSIM


class LossType(Enum):
    MSE = MSE
    threshold_wmse = ThresholdWMSE
    CRPS = CRPS
    SPL = SPL
    SSIM = SSIM
    exp_wmse = ExpWMSE
    expw_ssim = ExpwSSIM
