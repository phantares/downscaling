import torch
import torch.nn as nn
import numpy as np


class ZNormalizer(nn.Module):
    def __init__(self, mean: list, std: list) -> None:
        super().__init__()

        self.register_buffer(
            "mean", self._expand_array(torch.from_numpy(np.load(mean[0])))
        )
        self.register_buffer(
            "std", self._expand_array(torch.from_numpy(np.load(std[0])))
        )

        if len(mean) == 2:
            self.register_buffer(
                "mean_upper", self._expand_array(torch.from_numpy(np.load(mean[1])))
            )
            self.register_buffer(
                "std_upper", self._expand_array(torch.from_numpy(np.load(std[1])))
            )

    def _expand_array(self, data):
        return data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def standardize(self, data: torch.Tensor, is_upper: bool = False) -> torch.Tensor:
        if is_upper:
            return (data - self.mean_upper) / self.std_upper
        else:
            C = data.size(1)

            return (
                data
                - self.mean[
                    :,
                    :C,
                ]
            ) / self.std[
                :,
                :C,
            ]

    def inverse(self, data: torch.Tensor, is_upper: bool = False) -> torch.Tensor:
        if is_upper:
            return data * self.std_upper + self.mean_upper
        else:
            C = data.size(1)
            return (
                data
                * self.std[
                    :,
                    :C,
                ]
                + self.mean[
                    :,
                    :C,
                ]
            )
