from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from ...utils import MapPlotter


class FigureLogger(Callback):
    def __init__(self, log_val_batch_index=[1, 13, 16, 19], log_val_epoch=1):
        self.train_batch = None
        self.val_batch_index = log_val_batch_index
        self.val_epoch = log_val_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        if batch_index == 0:
            self.train_batch = batch

    def on_train_epoch_end(self, trainer, pl_module):
        if self.train_batch is None:
            return

        with torch.no_grad():
            batch = self.train_batch

            if len(batch) == 2:
                input, target = batch
                output = pl_module(input)
            elif len(batch) == 3:
                input_surface, input_upper, target = batch
                output = pl_module(input_surface, input_upper)

            if pl_module.scaling:
                output = pl_module.rain_scaling.inverse(output)

        self.log_tensorboard_image(trainer, f"train/target", target[0, 0])
        self.log_tensorboard_image(trainer, f"train/output", output[0, 0])

        self.train_batch = None

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_index):
        if batch_index not in self.val_batch_index:
            return

        with torch.no_grad():
            if len(batch) == 2:
                input, target = batch
                output = pl_module(input)
            elif len(batch) == 3:
                input_surface, input_upper, target = batch
                output = pl_module(input_surface, input_upper)

            if pl_module.scaling:
                output = pl_module.rain_scaling.inverse(output)

        case = f"case_{batch_index}"
        if trainer.current_epoch == self.val_epoch:
            self.log_tensorboard_image(trainer, f"{case}/target", target[0, 0])
        self.log_tensorboard_image(trainer, f"{case}/output", output[0, 0])

    def log_tensorboard_image(self, trainer, title: str, data):
        logger = self.get_tensorboard_logger(trainer)

        plotter = MapPlotter(data.detach().cpu())
        logger.add_figure(
            title,
            plotter.plot_map(),
            global_step=trainer.current_epoch,
        )

    def get_tensorboard_logger(self, trainer):
        for logger in trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tensorboard_logger = logger.experiment
                return tensorboard_logger

        raise ValueError("TensorboardLogger not found in trainer")
