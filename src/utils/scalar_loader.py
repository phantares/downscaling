from datetime import datetime
import calendar
import numpy as np

from .constants import SECONDS_IN_DAY, SECONDS_IN_HOUR, SECONDS_IN_MINUTE


class TimeLoader:
    def __init__(self, files, time_name: str = "Coordinates/time") -> None:
        times = []
        for file in files:
            times.append(file[time_name])
        self.times = np.concatenate(times)

    def load(self) -> np.ndarray:
        doy_sin = []
        doy_cos = []
        tod_sin = []
        tod_cos = []

        for time in self.times:
            time = datetime.fromisoformat(time.decode("utf-8"))

            days_in_year = 366 if calendar.isleap(time.year) else 365
            doy = int(time.strftime("%j"))
            doy_sin.append(np.sin(2 * np.pi * doy / days_in_year))
            doy_cos.append(np.cos(2 * np.pi * doy / days_in_year))

            tod = (
                time.hour * SECONDS_IN_HOUR
                + time.minute * SECONDS_IN_MINUTE
                + time.second
            )
            tod_sin.append(np.sin(2 * np.pi * tod / SECONDS_IN_DAY))
            tod_cos.append(np.cos(2 * np.pi * tod / SECONDS_IN_DAY))

        return (
            np.array(doy_sin, dtype=np.float32),
            np.array(doy_cos, dtype=np.float32),
            np.array(tod_sin, dtype=np.float32),
            np.array(tod_cos, dtype=np.float32),
        )
