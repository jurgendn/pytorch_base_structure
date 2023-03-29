import torch
from torch import Tensor, nn


class UselessLayer(nn.Module):

    def __init__(self) -> None:
        super(UselessLayer, self).__init__()
        self.seq = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.seq(x)
        return x