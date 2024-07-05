import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import h5py

from src.data_loaders import Hdf5Loader
from src.model_architectures import SimpleFramework, GanFramework


def main(path: str) -> None:
    cfg = OmegaConf.load(Path("experiments", path, ".hydra", "config.yaml"))

    checkpoints = list(Path("checkpoints", path).glob("*.ckpt"))
    best_model = find_best_model(checkpoints)

    framework = cfg.model.framework
    match framework:
        case "simple":
            framework = SimpleFramework
        case "gan":
            framework = GanFramework
    model = framework.load_from_checkpoint(best_model)

    data_loader = Hdf5Loader(cfg.dataset, **cfg.trainer.data_config)

    logger = TensorBoardLogger(save_dir=Path("logs", path), name="", version="test")

    trainer = Trainer(
        logger=logger,
        accelerator="cpu",
    )
    trainer.test(model, data_loader)

    input_file = str(cfg.dataset.test[0])
    output_file = f"{path.replace('/', '_')}_{input_file.split('_')[-2]}_{input_file.split('_')[-1]}"
    output_file = Path(input_file).parent / output_file

    with h5py.File(input_file, "r") as input:
        with h5py.File(output_file, "w") as output:
            for key in input.keys():
                input.copy(key, output)
            output["Variables/precipitation"][:] = np.squeeze(model.test_outputs)


def find_best_model(checkpoints):
    for checkpoint in checkpoints:
        callbacks = torch.load(checkpoint)["callbacks"]

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
    args = parser.parse_args()

    main(args.path)
