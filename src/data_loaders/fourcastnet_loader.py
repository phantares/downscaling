import numpy as np

from ..file_readers import Hdf5Reader
from ..utils import get_lonlat_index, z_normalize, robust_scaling


class FourcastnetLoader:
    def __init__(
        self,
        file: str,
        scaling_path: str = "/wk1/pei/Fourcastnet/2021/statistics/",
        scaling_method: str = "robust",
    ):

        self.file = Hdf5Reader(file).file
        self.INPUT_NUMBERS = self.file["Attributes/inputs_number"][()] + 1

        datas = self._load_data()

        match scaling_method:
            case "robust":
                scaling_function = robust_scaling
            case "z_norm":
                scaling_function = z_normalize
        self.datas = scaling_function(datas, scaling_path, [slice(self.INPUT_NUMBERS)])

    def _load_data(self):
        latitude = np.array(self.file["Coordinates/latitude"])
        longitude = np.array(self.file["Coordinates/longitude"])
        latmin, latmax, lonmin, lonmax = get_lonlat_index(latitude, longitude)

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
