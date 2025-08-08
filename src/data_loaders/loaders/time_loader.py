from datetime import datetime
import numpy as np

from ...utils import encode_time_feature


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

            day_sin, day_cos, time_sin, time_cos = encode_time_feature(time)
            doy_sin.append(day_sin)
            doy_cos.append(day_cos)
            tod_sin.append(time_sin)
            tod_cos.append(time_cos)

        return (
            np.array(doy_sin, dtype=np.float32),
            np.array(doy_cos, dtype=np.float32),
            np.array(tod_sin, dtype=np.float32),
            np.array(tod_cos, dtype=np.float32),
        )
