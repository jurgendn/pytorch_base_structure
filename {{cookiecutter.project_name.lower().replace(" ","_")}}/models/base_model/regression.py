from abc import abstractmethod
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import functional as FM


class LightningRegression(LightningModule):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super(LightningRegression, self).__init__(*args, **kwargs)
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def loss(self, input: Tensor, output: Tensor, **kwargs):
        return 0

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self.forward(x)
        loss = self.loss(input=pred, target=y)
        mse = FM.mean_squared_error(preds=pred, target=y)
        mape = FM.mean_absolute_percentage_error(preds=pred, target=y)
        return {"loss": loss, "mse": mse, "mape": mape}

    @torch.no_grad
    def training_step_end(self, step_output):
        return step_output

    @torch.no_grad
    def training_epoch_end(self, outputs):

        def average(key: str) -> Tensor:
            target_arr = torch.Tensor([val[key] for val in outputs]).float()
            return target_arr.mean()

        epoch_loss = average('loss')
        epoch_mse = average('mse')
        epoch_mape = average('mape')

        self.log("training/loss", epoch_loss)
        self.log("training/MSE", epoch_mse)
        self.log("training/MAPE", epoch_mape)

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self.forward(x)
        loss = self.loss(input=pred, target=y)
        mse = FM.mean_squared_error(preds=pred, target=y)
        mape = FM.mean_absolute_percentage_error(preds=pred, target=y)
        return {"loss": loss, "mse": mse, "mape": mape}

    @torch.no_grad
    def validation_step_end(self, step_output):
        return step_output

    @torch.no_grad
    def validation_epoch_end(self, outputs):

        def average(key: str) -> Tensor:
            target_arr = torch.Tensor([val[key] for val in outputs]).float()
            return target_arr.mean()

        epoch_loss = average('loss')
        epoch_mse = average('mse')
        epoch_mape = average('mape')

        self.log("valid/loss", epoch_loss)
        self.log("valid/MSE", epoch_mse)
        self.log("valid/MAPE", epoch_mape)