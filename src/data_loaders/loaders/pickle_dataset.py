import torch
from torch.utils.data import Dataset


class FourcastnetPickleDataset(Dataset):

    def __init__(self, files: str, include_rain: bool = False):
        super().__init__()

        self.data = torch.load(files[0])
        self.target = torch.load(files[1])

        if not include_rain:
            self.data = self.data[
                :,
                :-1,
            ]

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


class PanguPickleDataset(Dataset):

    def __init__(self, files: str, include_rain: bool = False):
        super().__init__()

        self.surface = torch.load(files[0])
        self.upper = torch.load(files[1])
        self.target = torch.load(files[2])

        if not include_rain:
            self.surface = self.surface[
                :,
                :-1,
            ]

    def __len__(self):
        return self.surface.shape[0]

    def __getitem__(self, index: int):
        return (
            self.surface[index,],
            self.upper[index,],
            self.target[
                :,
                index,
            ],
        )
