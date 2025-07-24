import torch
import torch.nn as nn


class LogNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float32))

    def standardize(self, data: torch.Tensor) -> torch.Tensor:
        return torch.log(data / self.eps + 1)

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        return (torch.exp(data) - 1) * self.eps
