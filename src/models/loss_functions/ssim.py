import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class SSIM(nn.Module):
    def __init__(self, scale: float = 200):
        super().__init__()

        self.scale = scale
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, prediction, target):
        prediction = prediction / (prediction + self.scale)
        target = target / (target + self.scale)

        ssim = self.ssim.to(prediction.device)
        ssim = ssim(prediction, target)

        return 1 - ssim
