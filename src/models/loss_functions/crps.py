import torch
import torch.nn as nn


class CRPS(nn.Module):

    def __init__(self, window_size: int = 9, integral_number: int = 1000):
        super().__init__()

        self.avg = nn.AvgPool2d(
            window_size, stride=1, padding=window_size // 2, count_include_pad=False
        )
        self.number = integral_number

    def forward(self, prediction, target):
        prediction_disturbance = prediction - self.avg(prediction)
        target_disturbance = target - self.avg(target)

        B = prediction.size(0)

        prediction_flat = prediction_disturbance.view(B, -1)
        target_flat = target_disturbance.view(B, -1)

        x_min = torch.min(prediction_flat.min(dim=1)[0], target_flat.min(dim=1)[0])
        x_max = torch.max(prediction_flat.max(dim=1)[0], target_flat.max(dim=1)[0])
        x_norm = torch.linspace(0, 1, self.number, device=prediction.device).unsqueeze(
            0
        )
        x = x_norm * (x_max - x_min).unsqueeze(1) + x_min.unsqueeze(1)

        prediction_cdf = self._calculate_cdf(prediction_flat, x)
        target_cdf = self._calculate_cdf(target_flat, x)

        diff_sq = (prediction_cdf - target_cdf) ** 2
        crps = torch.trapz(diff_sq, x, dim=1)

        return torch.mean(crps)

    def _calculate_cdf(self, data, thresholds):
        data = data.unsqueeze(1)
        thresholds = thresholds.unsqueeze(2)
        cdf = (data <= thresholds).float().mean(dim=2)

        return cdf


class PatchCRPS(nn.Module):

    def __init__(self, window_size: int = 9):
        super().__init__()

        self.avg = nn.AvgPool2d(
            window_size, stride=1, padding=window_size // 2, count_include_pad=False
        )
        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2)

    def forward(self, prediction, target):
        prediction_disturbance = prediction - self.avg(prediction)
        target_disturbance = target - self.avg(target)

        prediction_patch = self.unfold(prediction_disturbance).transpose(1, 2)
        target_patch = self.unfold(target_disturbance).transpose(1, 2)

        prediction_sort, _ = torch.sort(prediction_patch, dim=-1)
        target_sort, _ = torch.sort(target_patch, dim=-1)

        crps = (prediction_sort - target_sort) ** 2

        return torch.mean(crps)
