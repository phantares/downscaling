import torch
import torch.nn as nn
import numpy as np


class RobustScaler(nn.Module):
    def __init__(self, med: list, iqr: list) -> None:
        super().__init__()

        self.register_buffer(
            "med", self._expand_array(torch.from_numpy(np.load(med[0])))
        )
        self.register_buffer(
            "iqr", self._expand_array(torch.from_numpy(np.load(iqr[0])))
        )

        if len(med) == 2:
            self.register_buffer(
                "med_upper", self._expand_array(torch.from_numpy(np.load(med[1])))
            )
            self.register_buffer(
                "iqr_upper", self._expand_array(torch.from_numpy(np.load(iqr[1])))
            )

    def _expand_array(self, data):
        return data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def standardize(self, data: torch.Tensor, is_upper: bool = False) -> torch.Tensor:
        if is_upper:
            return (data - self.med_upper) / self.iqr_upper
        else:
            C = data.size(1)

            return (
                data
                - self.med[
                    :,
                    :C,
                ]
            ) / self.iqr[
                :,
                :C,
            ]

    def inverse(self, data: torch.Tensor, is_upper: bool = False) -> torch.Tensor:
        if is_upper:
            return data * self.iqr_upper + self.med_upper
        else:
            C = data.size(1)
            return (
                data
                * self.iqr[
                    :,
                    :C,
                ]
                + self.med[
                    :,
                    :C,
                ]
            )
