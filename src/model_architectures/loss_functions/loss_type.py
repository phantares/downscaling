from enum import Enum

from .weightedMSE import WeightedMSE

class LossType(Enum):
   weightedMSE = WeightedMSE