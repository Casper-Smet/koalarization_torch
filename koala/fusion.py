"""Fusion layer architecture."""


import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 256, kernal_size=1)
        self.act = nn.ReLU()
    
    def forward(self, img: torch.Tensor, feat_vec: torch.Tensor):
        ...
