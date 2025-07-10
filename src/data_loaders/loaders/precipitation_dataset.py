from torch.utils.data import Dataset
import numpy as np

from ...file_readers import Hdf5Reader
from ...utils import TimeLoader


class PrecipitationDataset(Dataset):

    def __init__(
        self,
        files: list[str],
        variables_name: list[str] = [
            "Variables/precipitation",
        ],
        targets_name: list[str] = ["Variables/target"],
        add_time: bool = False,
    ):

        super().__init__()

        self.files = []
        for file in files:
            reader = Hdf5Reader(file)
            self.files.append(reader.file)

        self.datas, self.targets = self._stack_data(variables_name, targets_name)

        self.scalars = []
        if add_time:
            time_scalars = TimeLoader(files=self.files).load()
            self.scalars.extend(time_scalars)
        self.scalars = np.array(self.scalars, dtype=np.float32)

    def __len__(self):
        length = 0
        for file in self.files:
            length += file["Coordinates/time"].shape[0]

        return length

    def __getitem__(self, index: int):
        input = {
            "input_surface": self.datas[
                :,
                index,
            ]
        }

        if len(self.scalars) > 0:
            input["input_scalar"] = self.scalars[:, index]

        return (
            input,
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
                [file[variable] for variable in variables_name],
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
