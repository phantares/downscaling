from pathlib import Path
from omegaconf import DictConfig
import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from src.data_loaders import Hdf5Loader
from src.model_architectures import SimpleFramework, GanFramework


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

    framework = cfg.model.framework
    del cfg.model.framework
    match framework:
        case "simple":
            framework = SimpleFramework
        case "gan":
            framework = GanFramework

    model = framework(
        hydra_config.runtime.choices.model.removesuffix("_gan"),
        cfg.model,
        hydra_config.runtime.choices["trainer/loss"],
        cfg.trainer.loss,
        hydra_config.runtime.choices["trainer/optimizer"],
        cfg.trainer.optimizer.config,
        cfg.trainer.optimizer.lr_scheduler,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=Path("checkpoints", experiment_name, sub_experiment_name),
            filename="{epoch}-{step}-{val_loss:.6f}",
            monitor="val_loss",
            save_top_k=3,
            save_last=True,
            mode="min",
        )
    ]

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if cfg.trainer.early_stopping:
        callbacks.append(EarlyStopping(**cfg.trainer.early_stopping))

    trainer = Trainer(
        benchmark=True,
        logger=logger,
        **cfg.trainer.max_settings,
        accelerator="gpu",
        callbacks=callbacks,
    )
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main()
