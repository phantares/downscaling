import numpy as np
from pathlib import Path

from ..file_readers import Hdf5Reader
from ..utils import get_lonlat_index


class FourcastnetLoader:
    def __init__(
        self,
        file: str,
        normalize_path: str = "/wk1/pei/Fourcastnet/2021/statistics/",
    ):

        self.file = Hdf5Reader(file).file

        datas = self._load_data()
        self.datas = self._normalize_data(datas, normalize_path)

    def _load_data(self):
        latitude = np.array(self.file["Coordinates/latitude"])
        longitude = np.array(self.file["Coordinates/longitude"])
        latmin, latmax, lonmin, lonmax = get_lonlat_index(latitude, longitude)

        self.INPUT_NUMBERS = self.file["Attributes/inputs_number"][()]
        datas = [[] for _ in range(self.INPUT_NUMBERS)]

        for variable_name in self.file["Variables"].keys():
            indexes = self.file[f"Variables/{variable_name}"].attrs["index"]
            data = np.nanmean(
                np.array(self.file[f"Variables/{variable_name}"])[
                    1:, ..., latmin : latmax - 1 : -1, lonmin : lonmax + 1
                ],
                axis=0,
            )

            if isinstance(indexes, np.int64):
                if indexes < self.INPUT_NUMBERS:
                    datas[indexes].append(data)

            else:
                for i, index in enumerate(indexes):
                    if index < self.INPUT_NUMBERS:
                        datas[index].append(data[i,])

        return np.squeeze(datas)

    def _normalize_data(self, datas: np.array, normalize_path: str):
        data_shape = np.shape(datas)

        mean = np.broadcast_to(
            np.load(str(Path(normalize_path, "mean.npy")))[
                : self.INPUT_NUMBERS, np.newaxis, np.newaxis
            ],
            data_shape,
        )
        std = np.broadcast_to(
            np.load(str(Path(normalize_path, "std.npy")))[
                : self.INPUT_NUMBERS, np.newaxis, np.newaxis
            ],
            data_shape,
        )

        return np.array((datas - mean) / std, dtype=np.float32)
