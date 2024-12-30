from pathlib import Path
import numpy as np


def z_normalize(data: np.array, file_path: str, array_slice: list, suffix: str = ""):
    data_shape = np.shape(data)

    data_index = tuple([*array_slice, np.newaxis, np.newaxis])
    mean = np.broadcast_to(
        np.load(str(Path(file_path, f"mean{suffix}.npy")))[data_index], data_shape
    )
    std = np.broadcast_to(
        np.load(str(Path(file_path, f"std{suffix}.npy")))[data_index], data_shape
    )

    return np.array((data - mean) / std, dtype=np.float32)


def robust_scaling(data: np.array, file_path: str, array_slice: list, suffix: str = ""):
    data_shape = np.shape(data)

    data_index = tuple([*array_slice, np.newaxis, np.newaxis])
    med = np.broadcast_to(
        np.load(str(Path(file_path, f"med{suffix}.npy")))[data_index], data_shape
    )
    iqr = np.broadcast_to(
        np.load(str(Path(file_path, f"iqr{suffix}.npy")))[data_index], data_shape
    )

    return np.array((data - med) / iqr, dtype=np.float32)
