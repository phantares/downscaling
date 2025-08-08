import sys
from pathlib import Path
import argparse
import onnxruntime as ort
from omegaconf import OmegaConf
from datetime import datetime, timedelta
import numpy as np
import h5py
from scipy.interpolate import griddata

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.file_readers import Hdf5Reader
from src.utils import encode_time_feature
from src.data_loaders import HOURS_PER_STEP
from src.utils.constants import HOURS_IN_DAY, M_TO_MM


def main(path: str):
    dwp = path.split("/")[0]
    sub_name = path.split("/")[1]

    model = Path("checkpoints", path, f"{dwp}_{sub_name}.onnx")
    ort_session = ort.InferenceSession(model)

    sample_file = OmegaConf.load(
        Path(
            "experiments",
            "configs",
            "data",
            "dataset",
            f"{dwp}_24hr_sg.yaml",
        )
    ).files.test[0]
    input_file = Hdf5Reader(sample_file)

    times = input_file.read_variable("Coordinates/time")
    times = [datetime.fromisoformat(time.decode("utf-8")) for time in times]

    target_lat = np.array(input_file.read_variable("Coordinates/latitude"))
    target_lon = np.array(input_file.read_variable("Coordinates/longitude"))
    target_lon, target_lat = np.meshgrid(target_lon, target_lat)

    cfg = OmegaConf.load(Path("experiments", dwp, sub_name, ".hydra", "config.yaml"))
    add_time = OmegaConf.select(
        cfg, "data.dataset.loader.config.add_time", default=False
    )

    steps = HOURS_IN_DAY // HOURS_PER_STEP
    rain_outputs = []
    for time in times:
        t = time - timedelta(hours=HOURS_PER_STEP)
        file = Path(
            "/wk1/pei/",
            dwp,
            str(t.year),
            t.strftime("%m"),
            f"{dwp}_0p25_{t.strftime('%Y%m%d%H')}.h5",
        )

        reader = Hdf5Reader(str(file))
        lat = np.array(reader.read_variable("Coordinates/latitude"))
        lon = np.array(reader.read_variable("Coordinates/longitude"))
        lon, lat = np.meshgrid(lon, lat)
        original_grid = np.transpose([lon.flatten(), lat.flatten()])

        rain = reader.read_variable("Variables/precip")[1 : 1 + steps]
        rain[rain < 0] = 0

        rain_6hr = []
        time_scalars = []
        for t in range(steps):
            data = griddata(
                original_grid,
                np.reshape(rain[t], -1) * M_TO_MM,
                (target_lon, target_lat),
            )
            data[data < 0] = 0
            rain_6hr.append(data)

            if add_time:
                sub_time = datetime.fromisoformat(
                    reader.read_variable("Coordinates/time")[t + 1].decode("utf-8")
                )

                time_scalars.append(list(encode_time_feature(sub_time)))

        rain_6hr = np.array(rain_6hr, dtype=np.float32)[
            :,
            np.newaxis,
        ]
        input = {"input_surface": rain_6hr}
        if add_time:
            time_scalars = np.array(time_scalars, dtype=np.float32)
            input["input_scalar"] = time_scalars

        rain_output = np.squeeze(ort_session.run(None, input))
        rain_outputs.append(np.sum(rain_output, axis=0))

    output_file = f"{path.replace('/', '_')}_{sample_file.split('_')[-2]}_{sample_file.split('_')[-1]}"
    output_file = Path(sample_file).parent / output_file
    with h5py.File(sample_file, "r") as input:
        with h5py.File(output_file, "w") as output:
            for key in input.keys():
                input.copy(key, output)
            output["Variables/precipitation"][:] = np.squeeze(rain_outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        type=str,
        help="Enter path.",
    )

    args = parser.parse_args()

    main(args.path)
