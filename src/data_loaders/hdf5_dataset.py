from torch.utils.data import Dataset
import numpy as np
from src.file_readers.hdf5_reader import Hdf5Reader


class Hdf5Dataset(Dataset):

    def __init__(
        self,
        files: str,
        variables_name: list[str] = [
            "Variables/precipitation",
        ],
        targets_name: list[str] = ["Variables/target"],
    ):

        super().__init__()

        self.files = []
        for file in files:
            reader = Hdf5Reader(file)
            self.files.append(reader.file)

        self.datas, self.targets = self._stack_data(variables_name, targets_name)

    def __len__(self):
        length = 0
        for file in self.files:
            length += file["Coordinates/time"].shape[0]

        return length

    def __getitem__(self, index: int):
        return (
            self.datas[
                :,
                index,
            ],
            self.targets[
                :,
                index,
            ],
        )

    def _stack_data(self, variables_name, targets_name):
        datas = []
        targets = []

        for file in self.files:
            data = np.stack(
                [np.sum(file[variable], axis=-1) for variable in variables_name],
                axis=0,
            )
            datas.append(data)

            target = np.stack(
                [file[variable] for variable in targets_name],
                axis=0,
            )
            targets.append(target)

        return (
            np.concatenate(datas, axis=1, dtype=np.float32),
            np.concatenate(targets, axis=1, dtype=np.float32),
        )
