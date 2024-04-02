from typing import List, Literal

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

    learning_rate_monitor_logging_interval: Literal["step", "epoch"] = Field(
        default="epoch"
    )


class ModelConfig(BaseModel):
    channels: List[int]
    kernel: List[int]
    n_classes: int


class OptimizerConfig(BaseModel):
    optimizer: str = Field(default="Adam")
    lr: float = Field(default=1e-3)
    weight_decay: float = Field(default=0.0)

    lr_scheduler_mode: Literal["min", "max"] = "min"
    lr_scheduler_factor: float = 0.2
    lr_scheduler_patience: int = 5
    lr_scheduler_threshold: float = 1e-4
    lr_scheduler_threshold_mode: str = "rel"
    lr_scheduler_min_lr: float = 0
    lr_scheduler_eps: float = 1e-8


class LoggerConfig(BaseModel):
    save_dir: str = Field(default="lightning_logs/")
    project: str
