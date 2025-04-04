from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from ..architectures import ModelType
from ..loss_functions import LossType
from ...utils import get_scheduler_with_warmup, MapPlotter, set_scaling


class SimpleFramework(LightningModule):

    def __init__(self, model_configs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.scaling = False
        if "scaling" in model_configs.framework.config:
            if model_configs.framework.config.scaling.execute:
                self.scaling = True
                self.rain_scaling = set_scaling(model_configs.framework.config.scaling)

        self.model = ModelType[model_configs.architecture.name].value(
            **model_configs.architecture.config
        )
        self.loss = LossType[model_configs.loss.name].value(**model_configs.loss.config)
        self.optimizer = model_configs.optimizer

        self.test_outputs = []

    def forward(self, *input):
        return self.model(*input)

    def general_step(self, target, *inputs):
        if self.scaling:
            inputs = list(inputs)
            inputs[-1] = self.rain_scaling.standardize(inputs[-1])
            target = self.rain_scaling.standardize(target)

        output = self(*inputs)
        loss = self.loss(output, target)

        if self.scaling:
            output = self.rain_scaling.inverse(output)

        return loss, output

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer.name)(
            self.parameters(), **self.optimizer.config
        )

        if "lr_scheduler" in self.optimizer:
            lr_scheduler = get_scheduler_with_warmup(
                optimizer,
                total_steps=int(self.trainer.estimated_stepping_batches),
                **self.optimizer.lr_scheduler,
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
        if len(batch) == 2:
            input, target = batch
            loss, output = self.general_step(target, input)
        elif len(batch) == 3:
            input_surface, input_upper, target = batch
            loss, output = self.general_step(target, input_surface, input_upper)

        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if batch_index == 0:
            # self.log_tensorboard_image(f"train/input", input[0, 0])
            self.log_tensorboard_image(f"train/target", target[0, 0])
            self.log_tensorboard_image(f"train/output", output[0, 0])

        return loss

    def validation_step(self, batch, batch_index: int):
        if len(batch) == 2:
            input, target = batch
            loss, output = self.general_step(target, input)
        elif len(batch) == 3:
            input_surface, input_upper, target = batch
            loss, output = self.general_step(target, input_surface, input_upper)

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
                # self.log_tensorboard_image(f"{case}/input", input[0, 0])
                self.log_tensorboard_image(f"{case}/target", target[0, 0])

            self.log_tensorboard_image(f"{case}/output", output[0, 0])

        return loss

    def test_step(self, batch, batch_index: int):
        if len(batch) == 2:
            input, target = batch
            loss, output = self.general_step(target, input)
        elif len(batch) == 3:
            input_surface, input_upper, target = batch
            loss, output = self.general_step(target, input_surface, input_upper)

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
