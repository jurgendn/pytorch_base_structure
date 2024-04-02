from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class SequentialCNN(nn.Module):
    def __init__(self, channels: List[int], kernel_size: List[int]) -> None:
        super(SequentialCNN, self).__init__()
        assert len(channels) - 1 == len(kernel_size)
        module_list = []
        for c_in, c_out, k in zip(channels[:-1], channels[1:], kernel_size):
            module_list.append(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k)
            )
            module_list.append(nn.LeakyReLU())
        self.module_list = nn.ModuleList(modules=module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.module_list:
            x = m(x)
            x = F.leaky_relu(x)
        return x
