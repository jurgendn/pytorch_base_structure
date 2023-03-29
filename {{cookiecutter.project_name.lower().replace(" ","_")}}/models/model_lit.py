import gin
from torch import Tensor, nn, optim
from torch.nn import functional as F

from .base_model.classification import LightningClassification
from .modules.sample_torch_module import UselessLayer


@gin.configurable
class UselessClassification(LightningClassification):

    def __init__(self, n_class: int, lr: float, **kwargs) -> None:
        super(UselessClassification).__init__()
        self.save_hyperparameters()
        self.main = nn.Sequential(UselessLayer(), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)

    def loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input=input, target=target)

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=self.hparams.step_size,
                                              gamma=self.hparams.gamma)
        return optimizer, scheduler
