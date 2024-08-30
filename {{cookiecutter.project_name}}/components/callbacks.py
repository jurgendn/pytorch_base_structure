import os
from abc import abstractmethod
from typing import List, Literal

import torch
from torch import nn

from components.config import CallBacksConfig
from components.trainer import Trainer


class Callback:
    def on_train_begin(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_train_end(self, trainer: Trainer):
        pass

    @abstractmethod
    def on_epoch_begin(self, trainer: Trainer, epoch: int):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer, epoch: int):
        pass

    @abstractmethod
    def on_batch_begin(self, trainer: Trainer, batch):
        pass

    @abstractmethod
    def on_batch_end(self, trainer: Trainer, batch):
        pass


class ModelCheckpoint(Callback):
    def __init__(
        self,
        dirpath: str,
        filename: str,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_best_only: bool = True,
        period: int = 1,
        verbose: int = 0,
    ):
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, trainer, epoch: int):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            current_value = trainer.metrics[epoch].get(self.monitor)
            if current_value is None:
                print(
                    f"Warning: Can't find {self.monitor} in metrics. Skipping model checkpoint."
                )
                return

            if self.save_best_only:
                if (self.mode == "min" and current_value < self.best_value) or (
                    self.mode == "max" and current_value > self.best_value
                ):
                    self.best_value = current_value
                    self._save_model(trainer.model, epoch, current_value)
            else:
                self._save_model(trainer.model, epoch, current_value)

    def _save_model(self, model: nn.Module, epoch, value):
        os.makedirs(self.dirpath, exist_ok=True)
        filename = self.filename.format(epoch=epoch + 1, **{self.monitor: value})
        path = os.path.join(self.dirpath, filename)
        torch.save(obj=model.state_dict(), f=path)
        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}: saving model to {filename}")


def get_callbacks(config: CallBacksConfig) -> List:
    """
    get_callbacks Get Callbacks

    Args:
        config (CallBacksConfig): _description_

    Returns:
        List: _description_
    """
    model_checkpoint = ModelCheckpoint(
        dirpath=config.model_checkpoint_dirpath,
        filename=config.model_checkpoint_filename,
        monitor=config.model_checkpoint_monitor,
        mode=config.model_checkpoint_mode,
        save_best_only=config.model_checkpoint_save_best_only,
        period=config.model_checkpoint_period,
        verbose=config.model_checkpoint_verbose,
    )
    return [
        model_checkpoint,
    ]
