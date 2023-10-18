from abc import abstractmethod
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningModule
from torch import Tensor


class LightningRegression(LightningModule):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super(LightningRegression, self).__init__(*args, **kwargs)
        self.train_step_output: List[Dict] = []
        self.validation_step_output: List[Dict] = []
        self.log_value_list: List[str] = ["loss", "mse", "mape"]

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def loss(self, input: Tensor, output: Tensor, **kwargs):
        return 0

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def __average(self, key: str, outputs: List[Dict]) -> Tensor:
        target_arr = torch.Tensor([val[key] for val in outputs]).float()
        return target_arr.mean()

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        for key in self.log_value_list:
            val = self.__average(key=key, outputs=self.train_step_output)
            log_name = f"training/{key}"
            self.log(name=log_name, value=val)

    @torch.no_grad()
    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        for key in self.log_value_list:
            val = self.__average(key=key, outputs=self.validation_step_output)
            log_name = f"training/{key}"
            self.log(name=log_name, value=val)
