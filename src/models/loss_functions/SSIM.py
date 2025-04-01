import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(
        self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0
    ):
        super().__init__()

        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range

    def forward(self, prediction, target):
        if self.data_range == 1.0:
            prediction, target = self._converse_grayscale(prediction, target)

        ssim = self._calculate_ssim(prediction, target)

        return 1 - ssim

    def _calculate_ssim(self, image1, image2):
        _, _, H, W = image1.shape

        window = self._gaussian_window(self.window_size, self.sigma).to(image1.device)
        window = window.expand(image1.shape[1], 1, self.window_size, self.window_size)

        mu1 = F.conv2d(
            image1, window, padding=self.window_size // 2, groups=image1.shape[1]
        )
        mu2 = F.conv2d(
            image2, window, padding=self.window_size // 2, groups=image2.shape[1]
        )

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(
                image1 * image1,
                window,
                padding=self.window_size // 2,
                groups=image1.shape[1],
            )
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(
                image2 * image2,
                window,
                padding=self.window_size // 2,
                groups=image2.shape[1],
            )
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(
                image1 * image2,
                window,
                padding=self.window_size // 2,
                groups=image1.shape[1],
            )
            - mu1_mu2
        )

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )

        return ssim_map.mean()

    def _gaussian_window(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        return g.unsqueeze(0) * g.unsqueeze(1)

    def _converse_grayscale(self, prediction, target):
        data_max = torch.max(prediction.max(), target.max())
        data_min = torch.min(prediction.min(), target.min())

        prediction_gray = (prediction - data_min) / (data_max - data_min)
        target_gray = (target - data_min) / (data_max - data_min)

        return prediction_gray, target_gray
