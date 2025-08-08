from datetime import datetime
import calendar
import numpy as np

from .constants import SECONDS_IN_DAY, SECONDS_IN_HOUR, SECONDS_IN_MINUTE


def encode_time_feature(time: datetime):
    days_in_year = 366 if calendar.isleap(time.year) else 365
    doy = int(time.strftime("%j"))
    doy_sin = np.sin(2 * np.pi * doy / days_in_year)
    doy_cos = np.cos(2 * np.pi * doy / days_in_year)

    tod = time.hour * SECONDS_IN_HOUR + time.minute * SECONDS_IN_MINUTE + time.second
    tod_sin = np.sin(2 * np.pi * tod / SECONDS_IN_DAY)
    tod_cos = np.cos(2 * np.pi * tod / SECONDS_IN_DAY)

    return doy_sin, doy_cos, tod_sin, tod_cos
