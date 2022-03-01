"""Koalarization encoder architecture.

The encoder architecture uses "same" padding in combination with stride at some points.
This isn't officially supported by PyTorch, ergo we're using timm's implementation.
The timm implementation might not work exactly as Tensorflow's does.

"""

import torch
from timm.models.layers.conv2d_same import Conv2dSame
from torch import nn


class Encoder(nn.Module):
    """Encoder architecture for Koalarization.

    All layers use the ReLU activation function.

    layer   Kernels         Stride
    conv    64  × (3 × 3)   2 × 2
    conv    128 × (3 × 3)   1 × 1
    conv    128 × (3 × 3)   2 × 2
    conv    256 × (3 × 3)   1 × 1
    conv    256 × (3 × 3)   2 × 2
    conv    512 × (3 × 3)   1 × 1
    conv    512 × (3 × 3)   1 × 1
    conv    256 × (3 × 3)   1 × 1

    """

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.conv1 = Conv2dSame(1, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(1, 128, kernel_size=3, padding="same")
        self.conv3 = Conv2dSame(1, 128, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(1, 256, kernel_size=3, padding="same")
        self.conv5 = Conv2dSame(1, 256, kernel_size=3, stride=2)
        self.conv6 = nn.Conv2d(1, 512, kernel_size=3, padding="same")
        self.conv7 = nn.Conv2d(1, 512, kernel_size=3, padding="same")
        self.conv8 = nn.Conv2d(1, 256, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Image, (1, 224, 224)

        Returns:
            torch.Tensor: Output for Fusion layer
        """
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x))
        return x
