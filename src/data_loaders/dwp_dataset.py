from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime, timedelta

from ..file_readers import Hdf5Reader
from .fourcastnet_loader import FourcastnetLoader


class DwpDataset(Dataset):

    def __init__(
        self,
        model: str,
        files: str,
        targets_name: list[str] = ["Variables/target"],
        root_path: str = "/wk1/pei/",
    ):

        super().__init__()

        self.root_path = Path(root_path, model)
        self.model = model

        self.files = []
        for file in files:
            reader = Hdf5Reader(file)
            self.files.append(reader.file)

        self.targets = self._stack_data(targets_name)

    def __len__(self):
        length = 0
        for file in self.files:
            length += file["Coordinates/time"].shape[0]

        return length

    def __getitem__(self, index: int):
        match self.model:
            case "Fourcastnet":
                surfaces = FourcastnetLoader(self.dwp_files[index]).datas

                return (
                    surfaces,
                    self.targets[
                        :,
                        index,
                    ],
                )

    def _stack_data(self, targets_name, hours_every_step: int = 6):
        self.dwp_files = []
        targets = []

        for file in self.files:

            times = file["Coordinates/time"]
            for t in times:
                time = datetime.strptime(str(t), "b'%Y-%m-%dT%H:%M:%S'") - timedelta(
                    hours=hours_every_step
                )
                self.dwp_files.append(
                    self.root_path.joinpath(
                        time.strftime("%Y"),
                        time.strftime("%m"),
                        f'{self.model}_0p25_{time.strftime("%Y%m%d%H")}.h5',
                    )
                )

            target = np.stack(
                [file[variable] for variable in targets_name],
                axis=0,
            )
            targets.append(target)

        return np.concatenate(targets, axis=1, dtype=np.float32)
