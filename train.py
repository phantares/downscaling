from pathlib import Path
from omegaconf import DictConfig
import hydra
from lightning.pytorch.loggers import TensorBoardLogger

from src.data_loaders import DataManager
from src.models import ModelBuilder
from src.trainers import TrainerBuilder
from src.utils import add_checkpoint_save_path


@hydra.main(version_base=None, config_path="experiments/configs", config_name="train")
def main(cfg: DictConfig) -> None:
    hydra_config = hydra.core.hydra_config.HydraConfig.get()

    experiment_name = cfg.experiment.name
    sub_experiment_name = Path(hydra_config.runtime.output_dir).name
    cfg = add_checkpoint_save_path(f"{experiment_name}/{sub_experiment_name}", cfg)
    print(f"Training experiment: {experiment_name}/{sub_experiment_name}")

    data_loader = DataManager(cfg.data)

    model = ModelBuilder(cfg.model).get_model()

    logger = TensorBoardLogger(
        save_dir=Path("logs", experiment_name),
        name=sub_experiment_name,
    )

    trainer = TrainerBuilder(cfg.trainer.fit, logger).get_trainer()

    trainer.fit(model, data_loader)


if __name__ == "__main__":
    main()
