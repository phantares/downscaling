from lightning import LightningModule
import torch

from ..architectures import ModelType
from ..loss_functions import LossType
from ...utils import get_scheduler_with_warmup, set_scaling


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

    def forward(self, *inputs):
<<<<<<< Updated upstream
        if self.scaling:
            inputs = list(inputs)
            inputs[-1] = self.rain_scaling.standardize(inputs[-1])

        output = self.model(*inputs)

        if self.scaling:
            output = self.rain_scaling.inverse(output)

        return output

    def general_step(self, target, *inputs):
=======
>>>>>>> Stashed changes
        if self.scaling:
            inputs = list(inputs)
            inputs[-1] = self.rain_scaling.standardize(inputs[-1])

        output = self.model(*inputs)

        return output

    def general_step(self, target, *inputs):
        if self.scaling:
            target = self.rain_scaling.standardize(target)

        output = self(*inputs)
        loss = self.loss(output, target)

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
        *inputs, target = batch
        loss, output = self.general_step(target, *inputs)

        self.log(
            f"train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_index: int):
        *inputs, target = batch
        loss, output = self.general_step(target, *inputs)

        self.log(
            f"val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def test_step(self, batch, batch_index: int):
        *inputs, target = batch
        loss, output = self.general_step(target, *inputs)

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
