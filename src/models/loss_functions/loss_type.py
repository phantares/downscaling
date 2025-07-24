from enum import Enum

from .mse import MSE
from .threshold_wmse import ThresholdWMSE
from .crps import CRPS
from .spl import SPL
from .ssim import SSIM
from .exp_wmse import ExpWMSE
from .expw_ssim import ExpwSSIM


class LossType(Enum):
    mse = MSE
    threshold_wmse = ThresholdWMSE
    crps = CRPS
    spl = SPL
    ssim = SSIM
    exp_wmse = ExpWMSE
    expw_ssim = ExpwSSIM
