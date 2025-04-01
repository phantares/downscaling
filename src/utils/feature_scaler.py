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
    def __init__(
        self, data: torch.Tensor, file_path: str, array_slice: list, suffix: str = ""
    ) -> None:

        data_shape = data.shape
        data_index = tuple([*array_slice, None, None])

        self.med = torch.from_numpy(
            np.load(str(Path(file_path, f"med{suffix}.npy")))[data_index]
        ).expand(data_shape)

        self.iqr = torch.from_numpy(
            np.load(str(Path(file_path, f"iqr{suffix}.npy")))[data_index]
        ).expand(data_shape)

    def standardize(self, data) -> torch.Tensor:
        return (data - self.med) / self.iqr

    def inverse(self, data) -> torch.Tensor:
        return data * self.iqr + self.med


class LogNormalizer:
    def __init__(self, eps: float = 1e5) -> None:
        self.eps = eps

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data * self.eps + 1)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return (torch.exp(data) - 1) / self.eps


class ScalingMethod(Enum):
    log_norm = LogNormalizer


def set_scaling(scaling_configs):
    rain_scaling = ScalingMethod[scaling_configs["rain"].method].value(
        **scaling_configs["rain"].config
    )

    return rain_scaling
