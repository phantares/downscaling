import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import h5py

from src.data_loaders import DataManager
from src.models import ModelBuilder
from src.trainers import TrainerBuilder


def main(path: str, as_onnx: bool = False) -> None:
    cfg = OmegaConf.load(Path("experiments", path, ".hydra", "config.yaml"))

    checkpoints = list(Path("checkpoints", path).glob("epoch*.ckpt"))
    best_model = find_best_model(checkpoints)

    framework = ModelBuilder(cfg.model).get_framework()
    model = framework.load_from_checkpoint(best_model)

    data_loader = DataManager(cfg.data)

    logger = TensorBoardLogger(save_dir=Path("logs", path), name="", version="test")

    trainer = TrainerBuilder(cfg.trainer.test, logger).get_trainer()
    trainer.test(model, data_loader)

    input_file = str(cfg.data.dataset.files.test[-1])
    if ".pt" in input_file:
        input_file = str(cfg.data.dataset.sample)
    output_file = f"{path.replace('/', '_')}_{input_file.split('_')[-2]}_{input_file.split('_')[-1]}"
    output_file = Path(input_file).parent / output_file

    with h5py.File(input_file, "r") as input:
        with h5py.File(output_file, "w") as output:
            for key in input.keys():
                input.copy(key, output)
            output["Variables/precipitation"][:] = np.squeeze(model.test_outputs)

    if as_onnx:
        data_loader.setup("fit")
        input, _ = next(iter(data_loader.train_dataloader()))

        onnx_file = Path("checkpoints", path, f"{path.replace('/', '_')}.onnx")
        torch.onnx.export(
            model.cpu(),
            input.to("cpu"),
            onnx_file,
            export_params=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )


def find_best_model(checkpoints):
    for checkpoint in checkpoints:
        callbacks = torch.load(checkpoint, weights_only=False)["callbacks"]

        for key in callbacks.keys():
            if "ModelCheckpoint" in key:
                val_loss = callbacks[key]["current_score"].item()

                if "best_score" not in locals():
                    best_score = val_loss
                    best_model = checkpoint

                else:
                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = checkpoint

    return best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "path",
        type=str,
        help="Enter path.",
    )
    parser.add_argument(
        "--onnx",
        type=bool,
        default=False,
        help="Whether output as onnx.",
    )
    args = parser.parse_args()

    main(args.path, args.onnx)
