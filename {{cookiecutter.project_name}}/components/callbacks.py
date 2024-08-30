import os
from abc import abstractmethod
from typing import List, Literal

import torch
from torch import nn

from components.config import CallBacksConfig
from components.trainer import Trainer


class Callback:
    @abstractmethod
    def on_train_begin(self, trainer: Trainer):
        """
        on_train_begin

        Args:
            trainer (Trainer): _description_
        """

    @abstractmethod
    def on_train_end(self, trainer: Trainer):
        """
        on_train_end

        Args:
            trainer (Trainer): _description_
        """

    @abstractmethod
    def on_epoch_begin(self, trainer: Trainer, epoch: int):
        """
        on_epoch_begin

        Args:
            trainer (Trainer): _description_
            epoch (int): _description_
        """

    @abstractmethod
    def on_epoch_end(self, trainer: Trainer, epoch: int):
        """
        on_epoch_end

        Args:
            trainer (Trainer): _description_
            epoch (int): _description_
        """

    @abstractmethod
    def on_batch_begin(self, trainer: Trainer, batch):
        """
        on_batch_begin

        Args:
            trainer (Trainer): _description_
            batch (_type_): _description_
        """

    @abstractmethod
    def on_batch_end(self, trainer: Trainer, batch):
        """
        on_batch_end

        Args:
            trainer (Trainer): _description_
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """


class ModelCheckpoint(Callback):
    """
    A callback for saving model checkpoints during training.

    The `ModelCheckpoint` callback is used to save the model's state at specified intervals during training,
    based on predefined criteria. It helps in preserving the best model configuration based on validation
    performance and can also save models at regular intervals, like every epoch.

    Attributes:
        dirpath (str): Directory to save the model checkpoints.
        filename (str): Filename to save the model checkpoints. Can contain formatting options like
            `{epoch}` or `{step}` to include the epoch or step number in the filename.
        monitor (str): The metric to monitor for saving checkpoints. Common choices are `val_loss`,
            `val_accuracy`, etc.
        save_best_only (bool): If `True`, only the model that shows the best performance on the monitored
            metric will be saved. If `False`, all checkpoints will be saved.
        mode (str): One of `min`, `max`. In `min` mode, the model checkpoint is saved when the monitored
            metric has the minimum value. In `max` mode, the model is saved when the monitored metric has
            the maximum value. Typically, use `min` for loss metrics and `max` for accuracy metrics.
        save_weights_only (bool): If `True`, only the model weights will be saved (without optimizer state).
            If `False`, the entire model including optimizer state will be saved.
        verbose (bool): If `True`, prints messages when saving a checkpoint.

    Methods:
        on_epoch_end(epoch: int, logs: Dict[str, Any]) -> None:
            Checks the monitored metric at the end of each epoch and saves a checkpoint if the metric
            meets the specified conditions.
        
        on_train_end(logs: Dict[str, Any]) -> None:
            Called at the end of training to save the final model checkpoint, if applicable.
        
        save_checkpoint(epoch: int, logs: Dict[str, Any], filepath: str) -> None:
            Saves the model checkpoint to the specified filepath.
    """
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
