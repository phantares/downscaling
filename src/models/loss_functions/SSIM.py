import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


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

        ssim = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(
            prediction.device
        )
        ssim = ssim(prediction, target)

        return 1 - ssim

    def _converse_grayscale(self, prediction, target):
        data_max = torch.max(prediction.max(), target.max())
        data_min = torch.min(prediction.min(), target.min())

        prediction_gray = (prediction - data_min) / (data_max - data_min)
        target_gray = (target - data_min) / (data_max - data_min)

        return prediction_gray, target_gray
