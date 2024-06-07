from enum import Enum

from .weightedMSE import WeightedMSE
from .CRPS import CRPS
from .SPL import SPL


class LossType(Enum):
    weightedMSE = WeightedMSE
    CRPS = CRPS
    SPL = SPL
