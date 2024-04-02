from typing import List
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateFinder,
)
from components.config import CallBacksConfig


def get_callbacks(config: CallBacksConfig) -> List:
    early_stopping = EarlyStopping(
        monitor=config.early_stopping_monitor,
        min_delta=config.early_stopping_min_delta,
        patience=config.early_stopping_patience,
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=config.model_checkpoint_dirpath,
        filename=config.model_checkpoint_filename,
        monitor=config.model_checkpoint_monitor,
        save_last=config.model_checkpoint_save_last,
        mode=config.model_checkpoint_mode,
    )
    learning_rate_finder = LearningRateFinder(
        min_lr=config.learning_rate_finder_min_lr,
        max_lr=config.learning_rate_finder_max_lr,
        num_training_steps=config.learning_rate_finder_num_training_steps,
        mode=config.learning_rate_finder_mode,
        early_stop_threshold=config.learning_rate_finder_early_stop_threshold,
        update_attr=config.learning_rate_finder_update_attr,
        attr_name=config.learning_rate_finder_attr_name,
    )
    return [early_stopping, model_checkpoint, learning_rate_finder]
