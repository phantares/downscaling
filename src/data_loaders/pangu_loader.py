import numpy as np

from ..file_readers import Hdf5Reader
from ..utils import get_lonlat_index, RobustScaler, ZNormalizer, LogNormalizer
from .constants import HOURS_PER_STEP

SURFACE_VARIABLES = ["mslet", "10u", "10v", "2t", "precip"]
UPPER_VARIABLES = ["z", "q", "t", "u", "v"]


class PanguLoader:
    def __init__(
        self,
        file: str,
        accumulated_hour: int = 24,
        scaling_method: str | None = None,
        scaling_path: str = "",
    ):

        self.file = Hdf5Reader(file).file

        if scaling_method:
            match scaling_method:
                case "robust":
                    scaling_function = RobustScaler
                case "z_norm":
                    scaling_function = ZNormalizer
                case "log_norm":
                    scaling_function = LogNormalizer

        datas = self._load_data("surface", accumulated_hour)
        self.datas_surface = scaling_function(datas, scaling_path).standardize()
        datas = self._load_data("upper", accumulated_hour)
        self.datas_upper = scaling_function(
            datas, scaling_path, suffix="_upper"
        ).standardize()

    def _load_data(self, layer: str, accumulated_hour: int = 24):
        latitude = np.array(self.file["Coordinates/latitude"])
        longitude = np.array(self.file["Coordinates/longitude"])
        latmin, latmax, lonmin, lonmax = get_lonlat_index(latitude, longitude)

        datas = []
        for variable_name in globals()[f"{layer.upper()}_VARIABLES"]:
            data = np.array(self.file[f"Variables/{variable_name}"])[
                1 : 1 + accumulated_hour // HOURS_PER_STEP,
                ...,
                latmin : latmax - 1 : -1,
                lonmin : lonmax + 1,
            ]

            if variable_name == "precip":
                data_function = np.nansum
            else:
                data_function = np.nanmean
            data = data_function(data, axis=0)

            datas.append(data)

        return np.squeeze(datas)
