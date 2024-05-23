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
        self.optimizer = optimizer_config

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

        if stage == "val" and batch_index in range(1, 11, 2):
            self.log_tensorboard_images(
                batch_index,
                input[0, 0],
                target[0, 0],
                output[0, 0],
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optimizer["lr"],
            betas=(self.optimizer["betas1"], self.optimizer["betas2"]),
        )

        return optimizer

    def training_step(self, batch, batch_index: int):
        return self.general_step(batch, batch_index, "train")

    def validation_step(self, batch, batch_index: int):
        return self.general_step(batch, batch_index, "val")

    def test_step(self, batch, batch_index: int):
        return self.general_step(batch, batch_index, "test")

    def log_tensorboard_images(self, batch_index: int, *datas):
        logger = self.get_tensorboard_logger()

        data_label = ["input", "target", "output"]
        for i, data in enumerate(datas):
            plotter = MapPlotter(data)
            logger.add_figure(
                f"case_{batch_index}/{data_label[i]}",
                plotter.plot_map(),
                global_step=self.global_step,
            )

    def get_tensorboard_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tensorboard_logger = logger.experiment
                return tensorboard_logger

        raise ValueError("TensorboardLogger not found in trainer")
