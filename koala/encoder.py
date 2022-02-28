"""Koalarization encoder architecture."""

import torch
from torch import nn


class Encoder(nn.Module):
    """Encoder architecture for Koalarization.

    All layers use the ReLU activation function.

    layer   Kernels         Stride
    conv    64 × (3 × 3)    2 × 2
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
