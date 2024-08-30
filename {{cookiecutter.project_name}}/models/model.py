from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, resnet18, resnet34, resnet50
from torchvision.models._utils import IntermediateLayerGetter

from components.config import ModelConfig
from models.modules.utils import get_num_channels

BACKBONE = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "mobilenet_v3_large": mobilenet_v3_large,
}


class MNISTModel(nn.Module):
    def __init__(self, model_config: ModelConfig) -> None:
        super(MNISTModel, self).__init__()
        self.model_config = model_config
        self.n_classes = model_config.n_classes
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
