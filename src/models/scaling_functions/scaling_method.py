from enum import Enum

from .log_normalizer import LogNormalizer
from .linearizer import Linearizer
from .robust_scaler import RobustScaler
from .z_normalizer import ZNormalizer


class ScalingMethod(Enum):
    log_norm = LogNormalizer
    linear = Linearizer
    robust = RobustScaler
    z_norm = ZNormalizer
