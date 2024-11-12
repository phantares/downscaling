import torch
from torch.utils.data import Dataset


class PickleDataset(Dataset):

    def __init__(self, model: str, files: str):

        super().__init__()

        self.model = model

        match model:
            case "Fourcastnet":
                self.data = torch.load(files[0])
                self.target = torch.load(files[1])

            case "Pangu":
                self.surface = torch.load(files[0])
                self.upper = torch.load(files[1])
                self.target = torch.load(files[2])

    def __len__(self):
        match self.model:
            case "Fourcastnet":
                return self.data.shape[0]
            case "Pangu":
                return self.surface.shape[0]

    def __getitem__(self, index: int):
        match self.model:
            case "Fourcastnet":
                return (
                    self.data[index,],
                    self.target[
                        :,
                        index,
                    ],
                )

            case "Pangu":
                return (
                    self.surface[index,],
                    self.upper[index,],
                    self.target[
                        :,
                        index,
                    ],
                )
