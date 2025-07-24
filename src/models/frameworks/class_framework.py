from lightning import LightningModule
import torch

from ..scaling_functions import ScalerLoader
from ..architectures import ModelType
from ..loss_functions import LossType
from ...utils import get_scheduler_with_warmup


class ClassFramework(LightningModule):

    def __init__(self, model_configs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.scaler = ScalerLoader(model_configs)
        self.model = ModelType[model_configs.architecture.name].value(
            **model_configs.architecture.config
        )
        self.loss = LossType[model_configs.loss.name].value(**model_configs.loss.config)
        self.class_loss = torch.nn.BCEWithLogitsLoss()
        self.class_weight = model_configs.framework.config.class_weight
        self.optimizer = model_configs.optimizer

        self.test_outputs = []

    def forward(self, **inputs):
        inputs = self.scaler.scale(**inputs)
        output, class_output = self.model(**inputs)

        return output, class_output

    def general_step(self, inputs, target, training: bool = False):
        target_class = (target > 0).float()
        if self.scaler.scale_rain:
            target = self.scaler.rain_scaler.standardize(target)

        output, class_output = self(**inputs)

        if training and self.trainer.current_epoch < 10:
            gated_output = output
        elif training and self.trainer.current_epoch < 20:
            gated_output = output * class_output.detach()
        else:
            gated_output = output * class_output

        class_loss = self.class_weight * self.class_loss(class_output, target_class)
        loss = self.loss(gated_output, target, (target_class == 1).float()) + class_loss

        return loss, output, class_output

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
        inputs, target = batch
        loss, output, class_output = self.general_step(inputs, target, training=True)

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
        inputs, target = batch
        loss, output, class_output = self.general_step(inputs, target)

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
        inputs, target = batch
        loss, output, class_output = self.general_step(inputs, target)
        output = output * (class_output > 0.5).float()

        self.log(
            f"test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.scaler.scale_rain:
            output = self.scaler.rain_scaler.inverse(output)

        self.test_outputs.append(output.cpu().numpy())

        return loss
