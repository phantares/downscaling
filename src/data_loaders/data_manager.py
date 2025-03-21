from lightning import LightningDataModule
from torch.utils.data import DataLoader

from .loaders import LoaderType


class DataManager(LightningDataModule):

    def __init__(self, data_configs) -> None:
        super().__init__()

        self.dataset = data_configs.dataset
        self.loader_configs = data_configs.config

    def setup(self, stage: str) -> None:
        loader = LoaderType[self.dataset.loader.name]

        self.train_dataset = loader.value(
            self.dataset.files.train, **self.dataset.loader.config
        )

        self.val_dataset = loader.value(
            self.dataset.files.val, **self.dataset.loader.config
        )

        self.test_dataset = loader.value(
            self.dataset.files.test, **self.dataset.loader.config
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.loader_configs.train)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.loader_configs.val)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.loader_configs.test)
