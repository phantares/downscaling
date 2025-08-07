import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data_preprocessors import PickleWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Enter model name of data.",
    )
    parser.add_argument(
        "-a",
        "--accumulated_hour",
        type=int,
        default=24,
        help="Enter accumulated hour for rain.",
    )

    args = parser.parse_args()

    PickleWriter(args.model, args.accumulated_hour)
