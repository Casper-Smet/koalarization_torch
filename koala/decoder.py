"""Koalarization decoder architecture."""

import torch
from torch import nn


class Decoder(nn.Module):
    """Decoder architecture for Koalarization.

    All layers use the ReLU activation function.
    layer   Kernels         Stride
    conv    128 × (3 × 3)   1 × 1
    upsamp      -           -
    conv    64  × (3 × 3)   1 × 1
    conv    64  × (3 × 3)   1 × 1
    upsamp      -           -
    conv    32  × (3 × 3)   1 × 1
    conv    2   × (3 × 3)   1 × 1
    upsamp      -           -


    """

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding="same")
        self.upsample = nn.UpsamplingNearest2d(size=2)
        self.conv2 = nn.Conv2d(128 * 2, 64, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding="same")
        self.conv4 = nn.Conv2d(64 * 2, 32, kernel_size=3, padding="same")
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, padding="same")
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Image, (1, 224, 224)

        Returns:
            torch.Tensor: Output for Fusion layer
        """
        x = self.act(self.conv1(x))
        x = self.upsample(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.upsample(x)
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.upsample()
        return x
