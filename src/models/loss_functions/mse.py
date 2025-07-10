import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target, mask=None):
        if mask is not None:
            mse = ((prediction - target) ** 2 * mask).sum() / (mask.sum() + 1e-6)
        else:
            mse = nn.MSELoss()(prediction, target)

        return mse
