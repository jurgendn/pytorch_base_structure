from torch import Tensor, nn, optim
from torch.nn import functional as F

from components.config import ModelConfig, OptimizerConfig
from models.base_model.classification import LightningClassification
from models.metrics.classification import classification_metrics
from torchvision.models._utils import IntermediateLayerGetter
from models.modules.utils import get_num_channels
from torchvision.models import resnet18, resnet34, resnet50, mobilenet_v3_large

BACKBONE = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "mobilenet_v3_large": mobilenet_v3_large,
}


class MNISTModel(LightningClassification):
    def __init__(
        self, model_config: ModelConfig, optimizer_config: OptimizerConfig
    ) -> None:
        super(MNISTModel, self).__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.n_classes = model_config.n_classes
        self.lr = self.optimizer_config.lr
        model_constructor = BACKBONE.get(model_config.backbone_name, None)
        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=(1, 1), padding=1
        )
        if model_constructor is None:
            raise ValueError("No model constructor")
        self.backbone = IntermediateLayerGetter(
            model=model_constructor(),
            return_layers=model_config.return_layers,
        )
        num_channels = get_num_channels(backbone=self.backbone, in_channels=3)
        self.fc = nn.Linear(
            in_features=num_channels, out_features=model_config.n_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.backbone(x)["output"]
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
            lr=self.lr,
            weight_decay=self.optimizer_config.weight_decay,
        )
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.optimizer_config.lr_scheduler_mode,
            factor=self.optimizer_config.lr_scheduler_factor,
            patience=self.optimizer_config.lr_scheduler_patience,
            threshold=self.optimizer_config.lr_scheduler_threshold,
            threshold_mode=self.optimizer_config.lr_scheduler_threshold_mode,
            min_lr=self.optimizer_config.lr_scheduler_min_lr,
            eps=self.optimizer_config.lr_scheduler_eps,
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "epoch", "monitor": "training/loss"}
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
