from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from src.model_architectures.models import ModelType
from src.model_architectures.lr_scheduler import get_scheduler_with_warmup
from src.model_architectures.loss_functions.loss_type import LossType
from src.figure_plotters.map_plotter import MapPlotter


class ModelFramework(LightningModule):

    def __init__(
        self,
        model_name,
        model_config,
        loss_name,
        loss_config,
        optimizer_name,
        optimizer_config,
        lr_scheduler: None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ModelType[model_name].value(**model_config)
        self.loss = LossType[loss_name].value(**loss_config)
        self.optimizer = optimizer_name
        self.optimizer_config = optimizer_config
        self.lr_scheduler = lr_scheduler

        self.test_outputs = []

    def forward(self, input):
        return self.model(input)

    def general_step(self, input, target):
        output = self(input)
        loss = self.loss(output, target)

        return loss, output

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(
            self.parameters(), **self.optimizer_config
        )

        if self.lr_scheduler:
            lr_scheduler = get_scheduler_with_warmup(
                optimizer,
                **self.lr_scheduler,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def training_step(self, batch, batch_index: int):
        input, target = batch
        loss, output = self.general_step(input, target)

        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_index == 0:
            self.log_tensorboard_image(f"train/input", input[0, 0])
            self.log_tensorboard_image(f"train/target", target[0, 0])
            self.log_tensorboard_image(f"train/output", output[0, 0])

        return loss

    def validation_step(self, batch, batch_index: int):
        input, target = batch
        loss, output = self.general_step(input, target)

        self.log(
            f"val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_index in [1, 13, 16, 19]:
            case = f"case_{batch_index}"

            if self.current_epoch == 1:
                self.log_tensorboard_image(f"{case}/input", input[0, 0])
                self.log_tensorboard_image(f"{case}/target", target[0, 0])

            self.log_tensorboard_image(f"{case}/output", output[0, 0])

        return loss

    def test_step(self, batch, batch_index: int):
        input, target = batch
        loss, output = self.general_step(input, target)

        self.log(
            f"test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.test_outputs.append(output.numpy())

        return loss

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
