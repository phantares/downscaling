from pathlib import Path
from omegaconf import DictConfig
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_loaders.hdf5_loader import Hdf5Loader
from src.model_architectures.model_framework import ModelFramework


@hydra.main(version_base=None, config_path="experiments/config", config_name="train")
def main(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()

    experiment_name = cfg.experiment.name
    sub_experiment_name = Path(hydra_config.runtime.output_dir).name
    print(f"Training sub-experiment: {sub_experiment_name}")

    logger = TensorBoardLogger(
        save_dir=Path("logs", experiment_name),
        name=sub_experiment_name,
    )

    data_loader = Hdf5Loader(cfg.dataset, **cfg.trainer.data_config)

    model = ModelFramework(
        hydra_config.runtime.choices.model,
        cfg.model,
        hydra_config.runtime.choices["trainer/loss"],
        cfg.trainer.loss,
        cfg.trainer.optimizer,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=Path("checkpoints", experiment_name, sub_experiment_name),
            filename="{epoch}-{val_loss_epoch:.6f}",
            monitor="val_loss_epoch",
            save_top_k=3,
            save_last=True,
            mode="min",
        )
    ]

    if cfg.trainer.early_stopping:
        callbacks.append(EarlyStopping(**cfg.trainer.early_stopping))

    trainer = Trainer(
        benchmark=True,
        logger=logger,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        devices=[1],
        callbacks=callbacks,
    )
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main()
