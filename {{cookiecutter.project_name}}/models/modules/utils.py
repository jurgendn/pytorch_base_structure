
import torch
from torch import nn


def get_num_channels(backbone: nn.Module, in_channels: int) -> int:
    x = torch.rand(size=(1, in_channels, 64, 64))
    out = backbone(x)
    num_channels = out.shape[1]
    return num_channels
