"""Koalarization decpder architecture."""

import torch
from torch import nn


class Decoder(nn.Module):
    """Decoder architecture for Koalarization.

    All layers use the ReLU activation function.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Image, (1, 224, 224)

        Returns:
            torch.Tensor: Output for Fusion layer
        """
        return x
