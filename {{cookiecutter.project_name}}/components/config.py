from typing import List

from pydantic import BaseModel, Field


class CallBacksConfig(BaseModel):
    early_stopping_monitor: str
    early_stopping_min_delta: int
    early_stopping_patience: int

    model_checkpoint_dirpath: str = Field(
        description="Save the weight", default="./checkpoints/"
    )
    model_checkpoint_mode: str
    model_checkpoint_filename: str
    model_checkpoint_monitor: str
    model_checkpoint_save_last: bool = Field(default=True)

    learning_rate_finder_min_lr: float
    learning_rate_finder_max_lr: float
    learning_rate_finder_num_training_steps: int
    learning_rate_finder_mode: str
    learning_rate_finder_early_stop_threshold: float
    learning_rate_finder_update_attr: bool = Field(default=True)
    learning_rate_finder_attr_name: str


class ModelConfig(BaseModel):
    channels: List[int]
    kernel: List[int]
    n_classes: int


class OptimizerConfig(BaseModel):
    optimizer: str = Field(default="Adam")
    lr: float = Field(default=1e-3)
    weight_decay: float = Field(default=0.0)

    lr_scheduler_gamma: float = 0.1
    lr_scheduler_last_epoch: int = -1
