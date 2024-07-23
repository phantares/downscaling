from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn

from .models import ModelType
from .loss_functions import LossType
from ..utils import get_scheduler_with_warmup, MapPlotter


class GanFramework(LightningModule):

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

        self.weight = model_config.discriminator_weight
        self.discriminator = ModelType["discriminator"].value(
            **model_config.discriminator
        )
        self.generator = ModelType[model_name].value(**model_config.generator)

        self.loss_g = LossType[loss_name].value(**loss_config)
        self.loss_d = nn.BCELoss()
        self.optimizer = optimizer_name
        self.optimizer_config = optimizer_config
        self.lr_scheduler = lr_scheduler

        self.automatic_optimization = False
        self.test_outputs = []

    def forward(self, input):
        return self.generator(input)

    def general_step(self, input, target):
        output = self(input)
        loss = self.loss_g(output, target)

        return loss, output

    def configure_optimizers(self):
        optimizer_g = getattr(torch.optim, self.optimizer)(
            self.generator.parameters(), **self.optimizer_config
        )
        optimizer_d = getattr(torch.optim, self.optimizer)(
            self.discriminator.parameters(), **self.optimizer_config
        )

        if self.lr_scheduler:
            get_scheduler_with_warmup(
                optimizer_g,
                total_steps=int(self.trainer.estimated_stepping_batches),
                **self.lr_scheduler,
            )

        return optimizer_g, optimizer_d

    def training_step(self, batch, batch_index: int):
        optimizer_g, optimizer_d = self.optimizers()
        input, target = batch
        batch_size = input.shape[0]

        self.toggle_optimizer(optimizer_g)
        loss_g, output = self.calculate_generator_loss(input, target, batch_size)
        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        loss_d = self.calculate_discriminator_loss(output, target, batch_size)
        optimizer_d.zero_grad()
        self.manual_backward(loss_d * self.weight)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.log_dict(
            {"generator_loss": loss_g, "discriminator_loss": loss_d},
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        if batch_index == 0:
            self.log_tensorboard_image(f"train/input", input[0, 0])
            self.log_tensorboard_image(f"train/target", target[0, 0])
            self.log_tensorboard_image(f"train/output", output[0, 0])

    def validation_step(self, batch, batch_index: int):
        input, target = batch
        batch_size = input.shape[0]

        loss_g, output = self.calculate_generator_loss(input, target, batch_size)
        loss_d = self.calculate_discriminator_loss(output, target, batch_size)

        self.log_dict(
            {"val_loss": loss_g, "val_loss_d": loss_d},
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

        return loss_g

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

    def calculate_generator_loss(self, input, target, batch_size):
        loss_pred, output_fake = self.general_step(input, target)

        output_disguise = self.discriminator(output_fake)
        loss_disguise = self.loss_d(
            output_disguise, torch.ones(batch_size, 1).type_as(target)
        )

        self.log_dict(
            {"prediction_loss": loss_pred, "disguise_loss": loss_disguise},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

        total_loss = loss_disguise * self.weight + loss_pred * (1 - self.weight)

        return total_loss, output_fake

    def calculate_discriminator_loss(self, output, target, batch_size):
        output_real = self.discriminator(target)
        loss_real = self.loss_d(output_real, torch.ones(batch_size, 1).type_as(target))
        output_fake = self.discriminator(output.detach())
        loss_fake = self.loss_d(output_fake, torch.zeros(batch_size, 1).type_as(target))

        total_loss = self.weight * (loss_fake + loss_real) / 2

        return total_loss

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
