import numpy as np
from pathlib import Path

from ..file_readers import Hdf5Reader
from ..utils import get_lonlat_index

SURFACE_VARIABLES = ["mslet", "10u", "10v", "2t"]
UPPER_VARIABLES = ["z", "q", "t", "u", "v"]


class PanguLoader:
    def __init__(
        self,
        file: str,
        normalize_path: str = "/wk1/pei/Pangu/2021/statistics/",
    ):

        self.file = Hdf5Reader(file).file

        datas = self._load_data("surface")
        self.datas_surface = self._normalize_data(datas, normalize_path, "surface")
        datas = self._load_data("upper")
        self.datas_upper = self._normalize_data(datas, normalize_path, "upper")

    def _load_data(self, layer: str):
        latitude = np.array(self.file["Coordinates/latitude"])
        longitude = np.array(self.file["Coordinates/longitude"])
        latmin, latmax, lonmin, lonmax = get_lonlat_index(latitude, longitude)

        datas = []
        for variable_name in globals()[f"{layer.upper()}_VARIABLES"]:
            data = np.nanmean(
                np.array(self.file[f"Variables/{variable_name}"])[
                    1:, ..., latmin : latmax - 1 : -1, lonmin : lonmax + 1
                ],
                axis=0,
            )
            datas.append(data)

        return np.squeeze(datas)

    def _normalize_data(self, datas: np.array, normalize_path: str, layer: str):
        data_shape = np.shape(datas)

        mean = np.broadcast_to(
            np.load(str(Path(normalize_path, f"mean_{layer}.npy")))[
                ..., np.newaxis, np.newaxis
            ],
            data_shape,
        )
        std = np.broadcast_to(
            np.load(str(Path(normalize_path, f"std_{layer}.npy")))[
                ..., np.newaxis, np.newaxis
            ],
            data_shape,
        )

        return np.array((datas - mean) / std, dtype=np.float32)
