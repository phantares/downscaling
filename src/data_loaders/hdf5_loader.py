from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .hdf5_dataset import Hdf5Dataset


class Hdf5Loader(LightningDataModule):

    def __init__(self, dataset: dict, **data_config) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_config = data_config

    def setup(self, stage: str) -> None:
        self.train_dataset = Hdf5Dataset(self.dataset.train)
        self.val_dataset = Hdf5Dataset(self.dataset.val)
        self.test_dataset = Hdf5Dataset(self.dataset.test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.data_config)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.data_config)

    def test_dataloader(self):
        self.data_config.pop("batch_size")

        return DataLoader(
            self.test_dataset, shuffle=False, batch_size=1, **self.data_config
        )
