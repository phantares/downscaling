import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    cycles: float = 0.5,
    last_epoch: int = -1,
):

    def cosine_decay(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.5, 0.5 * (1.0 + np.cos(np.pi * float(cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, cosine_decay, last_epoch)
