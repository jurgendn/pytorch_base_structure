from torch import Tensor, nn, optim
from torch.nn import functional as F

from .base_model.classification import LightningClassification
from .metrics.classification import classification_metrics
from .modules.sample_torch_module import UselessLayer


class UselessClassification(LightningClassification):

    def __init__(self, n_classes: int, lr: float, **kwargs) -> None:
        super(UselessClassification).__init__()
        self.save_hyperparameters()
        self.n_classes = n_classes
        self.lr = lr
        self.main = nn.Sequential(UselessLayer(), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input=input, target=target)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss(input=x, target=y)
        metrics = classification_metrics(preds=logits,
                                         target=y,
                                         num_classes=self.n_classes)

        self.train_batch_output.append({'loss': loss, **metrics})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss(input=x, target=y)
        metrics = classification_metrics(preds=logits,
                                         target=y,
                                         num_classes=self.n_classes)

        self.validation_batch_output.append({'loss': loss, **metrics})
        return loss
