from pathlib import Path
from enum import Enum
import torch
import numpy as np


class ZNormalizer:
    def __init__(
        self, data: torch.Tensor, file_path: str, array_slice: list, suffix: str = ""
    ) -> None:

        self.data = data
        data_shape = data.shape
        data_index = tuple([*array_slice, None, None])

        self.mean = torch.from_numpy(
            np.load(str(Path(file_path, f"mean{suffix}.npy")))[data_index]
        ).expand(data_shape)

        self.std = torch.from_numpy(
            np.load(str(Path(file_path, f"std{suffix}.npy")))[data_index]
        ).expand(data_shape)

    def standardize(self) -> torch.Tensor:
        return (self.data - self.mean) / self.std

    def inverse(self) -> torch.Tensor:
        return self.data * self.std + self.mean


class RobustScaler:
    def __init__(self, file_path: str, array_slice: list, suffix: str = "") -> None:
        data_index = tuple([*array_slice, None, None])

        self.med = torch.from_numpy(
            np.load(str(Path(file_path, f"med{suffix}.npy")))[data_index]
        )
        self.iqr = torch.from_numpy(
            np.load(str(Path(file_path, f"iqr{suffix}.npy")))[data_index]
        )

    def _expand_array(self, data_shape):
        med = self.med.expand(data_shape)
        iqr = self.iqr.expand(data_shape)

        return med, iqr

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        med, iqr = self._expand_array(data.shape)

        return (data - med) / iqr

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        med, iqr = self._expand_array(data.shape)

        return data * iqr + med


class Linearizer:
    def __init__(self, max_val, min_val=0) -> None:
        self.scale = max_val - min_val
        self.min = min_val

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.min) / self.scale

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.scale + self.min


class LogNormalizer:
    def __init__(self, eps: float = 1e-5) -> None:
        self.eps = eps

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data / self.eps + 1)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return (torch.exp(data) - 1) * self.eps


class ScalingMethod(Enum):
    log_norm = LogNormalizer
    linear = Linearizer


def set_scaling(scaling_configs):
    rain_scaling = ScalingMethod[scaling_configs["rain"].method].value(
        **scaling_configs["rain"].config
    )

    return rain_scaling
