import torch
import torch.nn as nn


class Linearizer(nn.Module):
    def __init__(self, max_val, min_val=0) -> None:
        super().__init__()

        self.register_buffer(
            "scale", torch.tensor(max_val - min_val, dtype=torch.float32)
        )
        self.register_buffer("min", torch.tensor(min_val, dtype=torch.float32))

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.min) / self.scale

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.scale + self.min
