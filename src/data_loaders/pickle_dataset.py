import torch
from torch.utils.data import Dataset
import numpy as np


class PickleDataset(Dataset):

    def __init__(self, files: str):

        super().__init__()

        self.data = torch.load(files[0])
        self.target = torch.load(files[1])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int):
        return (
            self.data[index,],
            self.target[
                :,
                index,
            ],
        )
