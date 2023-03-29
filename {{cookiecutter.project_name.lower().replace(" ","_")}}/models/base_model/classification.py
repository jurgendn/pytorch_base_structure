from abc import abstractmethod
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import functional as FM


class LightningClassification(LightningModule):

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super(LightningClassification, self).__init__(*args, **kwargs)
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass

    @abstractmethod
    def loss(self, input: Tensor, target: Tensor, **kwargs) -> Tensor:
        return 0

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self.forward(x)
        loss = self.loss(x=pred, y=y)
        f1 = FM.f1_score(preds=pred, target=y)
        recall = FM.recall(preds=pred, target=y)
        precision = FM.precision(preds=pred, target=y)
        return {
            "loss": loss,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }

    @torch.no_grad
    def training_step_end(self, step_output):
        return step_output

    @torch.no_grad
    def training_epoch_end(self, outputs):

        def average(key: str) -> Tensor:
            target_arr = torch.Tensor([val[key] for val in outputs]).float()
            return target_arr.mean()

        epoch_loss = average('loss')
        epoch_f1 = average('f1')
        epoch_recall = average('recall')
        epoch_precision = average('precision')

        self.log("training/loss", epoch_loss)
        self.log("training/f1", epoch_f1)
        self.log("training/recall", epoch_recall)
        self.log("training/precision", epoch_precision)

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self.forward(x)
        loss = F.cross_entropy(input=pred, target=y)
        f1 = FM.f1_score(preds=pred, target=y)
        recall = FM.recall(preds=pred, target=y)
        precision = FM.precision(preds=pred, target=y)
        return {
            "loss": loss,
            "f1": f1,
            "recall": recall,
            "precision": precision
        }
