from torch import Tensor, nn, optim
from torch.nn import functional as F

from components.config import ModelConfig, OptimizerConfig
from models.base_model.classification import LightningClassification
from models.metrics.classification import classification_metrics
from models.modules.commons import SequentialCNN
from models.modules.utils import get_num_channels


class MNISTModel(LightningClassification):
    def __init__(
        self, model_config: ModelConfig, optimizer_config: OptimizerConfig
    ) -> None:
        super(MNISTModel, self).__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.n_classes = model_config.n_classes
        self.backbone = SequentialCNN(
            channels=model_config.channels, kernel_size=model_config.kernel
        )
        num_channels = get_num_channels(
            backbone=self.backbone, in_channels=model_config.channels[0]
        )
        self.fc = nn.Linear(
            in_features=num_channels, out_features=model_config.n_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(input=x, output_size=(1, 1))
        x = x.view(size=(x.shape[0], -1))
        x = self.fc(x)
        x = F.leaky_relu(input=x)
        return x

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input=input, target=target)

    def configure_optimizers(self):
        optimizer_list = {"Adam": optim.Adam, "AdamW": optim.AdamW, "SGD": optim.SGD}
        caller = optimizer_list.get(self.optimizer_config.optimizer, optim.Adam)
        optimizer = caller(
            params=self.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.optimizer_config.lr_scheduler_gamma,
            last_epoch=self.optimizer_config.lr_scheduler_last_epoch,
        )
        return [optimizer], [
            {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
        ]

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss(input=logits, target=y)
        metrics = classification_metrics(
            preds=logits, target=y, num_classes=self.n_classes
        )

        self.train_batch_output.append({"loss": loss, **metrics})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss(input=logits, target=y)
        metrics = classification_metrics(
            preds=logits, target=y, num_classes=self.n_classes
        )

        self.validation_batch_output.append({"loss": loss, **metrics})
        return loss
