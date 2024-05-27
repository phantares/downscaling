from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from src.model_architectures.models import ModelType
from src.model_architectures.loss_functions.loss_type import LossType
from src.figure_plotters.map_plotter import MapPlotter


class ModelFramework(LightningModule):

    def __init__(
        self, model_name, model_config, loss_name, loss_config, optimizer_config
    ):
        super().__init__()

        self.model = ModelType[model_name].value(**model_config)
        self.loss = LossType[loss_name].value(**loss_config)
        self.optimizer_config = optimizer_config

    def forward(self):
        pass

    def general_step(self, batch, batch_index: int, stage: str):
        input, target = batch
        output = self.model(input)
        loss = self.loss(output, target)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if stage == "train" and batch_index == 0:
            self.log_tensorboard_image(f"{stage}/input", input[0, 0])
            self.log_tensorboard_image(f"{stage}/target", target[0, 0])
            self.log_tensorboard_image(f"{stage}/output", output[0, 0])

        if stage == "val" and batch_index in [1, 13, 16, 19]:
            case = f"case_{batch_index}"

            if self.current_epoch == 1:
                self.log_tensorboard_image(f"{case}/input", input[0, 0])
                self.log_tensorboard_image(f"{case}/target", target[0, 0])

            self.log_tensorboard_image(f"{case}/output", output[0, 0])

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            **self.optimizer_config,
        )

        return optimizer

    def training_step(self, batch, batch_index: int):
        return self.general_step(batch, batch_index, "train")

    def validation_step(self, batch, batch_index: int):
        return self.general_step(batch, batch_index, "val")

    def test_step(self, batch, batch_index: int):
        return self.general_step(batch, batch_index, "test")

    def log_tensorboard_image(self, title: str, data):
        logger = self.get_tensorboard_logger()

        plotter = MapPlotter(data)
        logger.add_figure(
            title,
            plotter.plot_map(),
            global_step=self.global_step,
        )

    def get_tensorboard_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tensorboard_logger = logger.experiment
                return tensorboard_logger

        raise ValueError("TensorboardLogger not found in trainer")
