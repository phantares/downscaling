from pathlib import Path
import numpy as np


class ZNormalizer:
    def __init__(
        self, data: np.ndarray, file_path: str, array_slice: list, suffix: str = ""
    ) -> None:

        self.data = data

        data_shape = np.shape(data)
        data_index = tuple([*array_slice, np.newaxis, np.newaxis])

        self.mean = np.broadcast_to(
            np.load(str(Path(file_path, f"mean{suffix}.npy")))[data_index], data_shape
        )
        self.std = np.broadcast_to(
            np.load(str(Path(file_path, f"std{suffix}.npy")))[data_index], data_shape
        )

    def standardize(self) -> np.ndarray:
        return np.array((self.data - self.mean) / self.std, dtype=np.float32)

    def inverse(self) -> np.ndarray:
        return np.array(self.data * self.std + self.mean, dtype=np.float32)


class RobustScaler:
    def __init__(
        self, data: np.ndarray, file_path: str, array_slice: list, suffix: str = ""
    ) -> None:

        self.data = data

        data_shape = np.shape(data)
        data_index = tuple([*array_slice, np.newaxis, np.newaxis])

        self.med = np.broadcast_to(
            np.load(str(Path(file_path, f"med{suffix}.npy")))[data_index], data_shape
        )
        self.iqr = np.broadcast_to(
            np.load(str(Path(file_path, f"iqr{suffix}.npy")))[data_index], data_shape
        )

    def standardize(self) -> np.ndarray:
        return np.array((self.data - self.med) / self.iqr, dtype=np.float32)

    def inverse(self) -> np.ndarray:
        return np.array(self.data * self.iqr + self.med, dtype=np.float32)


class LogNormalizer:
    def __init__(self, data: np.ndarray, eps: float = 1e5) -> None:
        self.data = data
        self.eps = eps

    def standardize(self) -> np.ndarray:
        return np.log(self.data * self.eps + 1)

    def inverse(self) -> np.ndarray:
        return (np.exp(self.data) - 1) / self.eps
