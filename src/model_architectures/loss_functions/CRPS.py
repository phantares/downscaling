import torch
import torch.nn as nn


class CRPS(nn.Module):

    def __init__(self, integral_number: int = 1000):
        super().__init__()

        self.number = integral_number

    def forward(self, prediction, target):
        return self._calculate_crps(torch.flatten(prediction), torch.flatten(target))

    def _calculate_crps(self, prediction, target):
        x = torch.linspace(
            float(min(torch.min(prediction), torch.min(target))),
            float(max(torch.max(prediction), torch.max(target))),
            self.number,
        )

        cdf_prediction = self._calculate_cdf(x, prediction).requires_grad_()
        cdf_target = self._calculate_cdf(x, target)
        diff = torch.abs(cdf_prediction - cdf_target)

        return torch.trapz(diff**2, x)

    def _calculate_cdf(self, x, data):
        cdf = [torch.mean((data <= value).float()) for value in x]
        return torch.tensor(cdf)


class L1CRPS(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        sort_p, _ = torch.sort(torch.flatten(prediction))
        sort_t, _ = torch.sort(torch.flatten(target))
        dx = torch.abs(sort_p - sort_t)
        loss = torch.sum(dx) / torch.numel(target)

        return loss
