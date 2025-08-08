import numpy as np

from ...file_readers import Hdf5Reader
from ...utils import get_lonlat_index
from ..configs import HOURS_PER_STEP


class FourcastnetLoader:
    def __init__(
        self,
        file: str,
        accumulated_hour: int = 24,
    ):

        self.file = Hdf5Reader(file).file
        self.INPUT_NUMBERS = self.file["Attributes/inputs_number"][()] + 1

        self.datas = self._load_data(accumulated_hour)

    def load_data(self):
        return self.datas

    def _load_data(self, accumulated_hour: int = 24):
        latitude = np.array(self.file["Coordinates/latitude"])
        longitude = np.array(self.file["Coordinates/longitude"])
        latmin, latmax, lonmin, lonmax = get_lonlat_index(latitude, longitude)

        datas = [[] for _ in range(self.INPUT_NUMBERS)]

        for variable_name in self.file["Variables"].keys():
            indexes = self.file[f"Variables/{variable_name}"].attrs["index"]

            data = np.array(self.file[f"Variables/{variable_name}"])[
                1 : 1 + accumulated_hour // HOURS_PER_STEP,
                ...,
                latmin : latmax - 1 : -1,
                lonmin : lonmax + 1,
            ]

            if variable_name == "precip":
                data[data < 0] = 0
                data_function = np.nansum
            else:
                data_function = np.nanmean
            data = data_function(data, axis=0)

            if isinstance(indexes, np.int64):
                if indexes < self.INPUT_NUMBERS:
                    datas[indexes].append(data)

            else:
                for i, index in enumerate(indexes):
                    if index < self.INPUT_NUMBERS:
                        datas[index].append(data[i,])

        return np.squeeze(datas)
