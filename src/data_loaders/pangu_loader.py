import numpy as np
from pathlib import Path

from ..file_readers import Hdf5Reader
from ..utils import get_lonlat_index, robust_scaling, z_normalize
from ..utils.feature_scaler import z_normalize

SURFACE_VARIABLES = ["mslet", "10u", "10v", "2t"]
UPPER_VARIABLES = ["z", "q", "t", "u", "v"]


class PanguLoader:
    def __init__(
        self,
        file: str,
        scaling_path: str = "/wk1/pei/Pangu/2021/statistics/",
        scaling_method: str = "robust",
    ):

        self.file = Hdf5Reader(file).file

        match scaling_method:
            case "robust":
                scaling_function = robust_scaling
            case "z_norm":
                scaling_function = z_normalize

        datas = self._load_data("surface")
        self.datas_surface = scaling_function(datas, scaling_path, [Ellipsis])
        datas = self._load_data("upper")
        self.datas_upper = scaling_function(
            datas, scaling_path, [Ellipsis, Ellipsis], "_upper"
        )

    def _load_data(self, layer: str):
        latitude = np.array(self.file["Coordinates/latitude"])
        longitude = np.array(self.file["Coordinates/longitude"])
        latmin, latmax, lonmin, lonmax = get_lonlat_index(latitude, longitude)

        datas = []
        for variable_name in globals()[f"{layer.upper()}_VARIABLES"][:2]:
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
