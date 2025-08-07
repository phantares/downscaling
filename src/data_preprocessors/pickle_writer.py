from pathlib import Path
from omegaconf import OmegaConf
import torch
from datetime import datetime, timedelta
import numpy as np

from ..data_loaders import (
    HOURS_PER_STEP,
    PrecipitationDataset,
    FourcastnetLoader,
    PanguLoader,
)


class PickleWriter:
    def __init__(
        self, model: str, accumulated_hour: int = 24, output_path: str = "/wk1/pei/"
    ):

        self.model = model
        self.root = Path(output_path, model)
        self.path = Path(output_path, model, f"{accumulated_hour}hr")

        self.cfg = OmegaConf.load(
            Path(
                "experiments",
                "configs",
                "data",
                "dataset",
                f"{model}_{accumulated_hour}hr_sg.yaml",
            )
        ).files

        for stage in ["train", "val", "test"]:
            self._write_file(model, stage, accumulated_hour)

    def _write_file(self, model: str, stage: str, accumulated_hour: int = 24) -> None:
        dataset = PrecipitationDataset(self.cfg[stage])

        target = dataset.targets
        torch.save(target, Path(self.path, f"{stage}_target.pt"))

        dwp_files = self._load_dwp_files(dataset.files, HOURS_PER_STEP)

        match model:
            case "Fourcastnet":
                loader = FourcastnetLoader

                datas = []
                for file in dwp_files:
                    datas.append(loader(file, accumulated_hour).load_data())

                torch.save(
                    np.array(datas, dtype=np.float32),
                    Path(self.path, f"{stage}_data.pt"),
                )

            case "Pangu":
                loader = PanguLoader

                datas_surface = []
                datas_upper = []
                for file in dwp_files:
                    datas = loader(file, accumulated_hour)
                    datas_surface.append(datas.datas_surface)
                    datas_upper.append(datas.datas_upper)

                torch.save(
                    np.array(datas_surface, dtype=np.float32),
                    Path(self.path, f"{stage}_surface.pt"),
                )
                torch.save(
                    np.array(datas_upper, dtype=np.float32),
                    Path(self.path, f"{stage}_upper.pt"),
                )

    def _load_dwp_files(self, files, hours_every_step: int = 6):
        dwp_files = []

        for file in files:

            times = file["Coordinates/time"]
            for t in times:
                time = datetime.strptime(str(t), "b'%Y-%m-%dT%H:%M:%S'") - timedelta(
                    hours=hours_every_step
                )

                dwp_files.append(
                    Path(
                        self.root,
                        time.strftime("%Y"),
                        time.strftime("%m"),
                        f'{self.model}_0p25_{time.strftime("%Y%m%d%H")}.h5',
                    )
                )

        return dwp_files
