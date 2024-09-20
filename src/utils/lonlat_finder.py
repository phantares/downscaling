import numpy as np


def get_lonlat_index(
    latitude,
    longitude,
    lat_min: float = 21,
    lat_max: float = 26.55,
    lon_min: float = 118.4,
    lon_max: float = 123.15,
) -> tuple[int, int, int, int]:

    latmin = np.argwhere(latitude <= lat_min)[0][0]
    latmax = np.argwhere(latitude >= lat_max)[-1][0]
    lonmin = np.argwhere(longitude <= lon_min)[-1][0]
    lonmax = np.argwhere(longitude >= lon_max)[0][0]

    return latmin, latmax, lonmin, lonmax
