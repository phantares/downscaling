from lightning import Trainer
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from .callbacks import FigureLoggerType


class TrainerBuilder:

    def __init__(self, trainer_configs, logger):

        callbacks = []
        if "callbacks" in trainer_configs:
            callbacks = self._load_callbacks(trainer_configs.callbacks)

        self.trainer = Trainer(
            logger=logger, callbacks=callbacks, **trainer_configs.config
        )

    def get_trainer(self):
        return self.trainer

    def _load_callbacks(self, callback_configs):
        callbacks = []

        callbacks.append(FigureLoggerType[callback_configs.figure]).value()

        if "checkpoint" in callback_configs:
            callbacks.append(ModelCheckpoint(**callback_configs.checkpoint))

        if "lr_monitor" in callback_configs:
            callbacks.append(LearningRateMonitor(**callback_configs.lr_monitor))

        if "early_stopping" in callback_configs:
            callbacks.append(EarlyStopping(**callback_configs.early_stopping))

        return callbacks
