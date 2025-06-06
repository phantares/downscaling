from lightning import LightningModule
import torch
import torch.nn as nn

from ..architectures import ModelType
from ..loss_functions import LossType
from ...utils import get_scheduler_with_warmup, set_scaling


class GanFramework(LightningModule):

    def __init__(self, model_configs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.scaling = False
        if "scaling" in model_configs.framework.config:
            if model_configs.framework.config.scaling.execute:
                self.scaling = True
                self.rain_scaling = set_scaling(model_configs.framework.config.scaling)

        self.weight = model_configs.framework.discriminator.weight
        self.discriminator = ModelType[
            model_configs.framework.discriminator.name
        ].value(**model_configs.framework.discriminator.config)
        self.generator = ModelType[model_configs.architecture.name].value(
            **model_configs.architecture.config
        )

        self.loss_g = LossType[model_configs.loss.name].value(
            **model_configs.loss.config
        )
        self.loss_d = nn.BCELoss()
        self.optimizer = model_configs.optimizer

        self.automatic_optimization = False
        self.test_outputs = []

    def forward(self, inputs):
        if self.scaling:
            inputs = list(inputs)
            inputs[-1] = self.rain_scaling.standardize(inputs[-1])

        output = self.generator(*inputs)

        return output

    def general_step(self, inputs, target):
        if self.scaling:
            target = self.rain_scaling.standardize(target)

        output = self(*inputs)
        loss = self.loss_g(output, target)

        return loss, output

    def configure_optimizers(self):
        optimizer_g = getattr(torch.optim, self.optimizer.name)(
            self.generator.parameters(), **self.optimizer.config
        )
        optimizer_d = getattr(torch.optim, self.optimizer.name)(
            self.discriminator.parameters(), **self.optimizer.config
        )

        if "lr_scheduler" in self.optimizer:
            get_scheduler_with_warmup(
                optimizer_g,
                total_steps=int(self.trainer.estimated_stepping_batches),
                **self.optimizer.lr_scheduler,
            )

        return optimizer_g, optimizer_d

    def training_step(self, batch, batch_index: int):
        optimizer_g, optimizer_d = self.optimizers()
        *inputs, target = batch
        batch_size = inputs[-1].shape[0]

        self.toggle_optimizer(optimizer_g)
        loss_g, output = self.calculate_generator_loss(inputs, target, batch_size)
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

    def validation_step(self, batch, batch_index: int):
        *inputs, target = batch
        batch_size = inputs[-1].shape[0]

        loss_g, output = self.calculate_generator_loss(inputs, target, batch_size)
        loss_d = self.calculate_discriminator_loss(output, target, batch_size)

        self.log_dict(
            {"val_loss": loss_g, "val_loss_d": loss_d},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss_g

    def test_step(self, batch, batch_index: int):
        *inputs, target = batch
        loss, output = self.general_step(inputs, target)

        self.log(
            f"test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.scaling:
            output = self.rain_scaling.inverse(output)

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
