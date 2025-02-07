from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .precipitation_dataset import PrecipitationDataset
from .dwp_dataset import DwpDataset
from .pickle_dataset import PickleDataset


class Hdf5Loader(LightningDataModule):

    def __init__(self, dataset: dict, **data_config) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_config = data_config

    def setup(self, stage: str) -> None:
        match self.dataset.dataset:
            case "single":
                self.train_dataset = PrecipitationDataset(
                    self.dataset.train, **self.dataset.scaling
                )
                self.val_dataset = PrecipitationDataset(
                    self.dataset.val, **self.dataset.scaling
                )
                self.test_dataset = PrecipitationDataset(
                    self.dataset.test, **self.dataset.scaling
                )

            case "multi":
                self.train_dataset = DwpDataset(self.dataset.name, self.dataset.train)
                self.val_dataset = DwpDataset(self.dataset.name, self.dataset.val)
                self.test_dataset = DwpDataset(self.dataset.name, self.dataset.test)

            case "pickle":
                self.train_dataset = PickleDataset(
                    self.dataset.name, self.dataset.train, self.dataset.include_rain
                )
                self.val_dataset = PickleDataset(
                    self.dataset.name, self.dataset.val, self.dataset.include_rain
                )
                self.test_dataset = PickleDataset(
                    self.dataset.name, self.dataset.test, self.dataset.include_rain
                )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.data_config)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.data_config)

    def test_dataloader(self):
        self.data_config.pop("batch_size")

        return DataLoader(
            self.test_dataset, shuffle=False, batch_size=1, **self.data_config
        )
