from typing import List

import torch
from torch import nn

from models.modules.commons import SequentialCNN


class SampleModel(nn.Module):
    def __init__(self, channels: List[int], kernel_size: List[int]) -> None:
        super(SampleModel, self).__init__()
        self.cnn_1 = SequentialCNN(channels=channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
