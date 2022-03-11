"""Fusion layer architecture."""


import torch
from torch import nn


class Fusion(nn.Module):
    """Fusion architecture for Koalarization.

    All layers use the ReLU activation function.
    
    Layer   Kernels         Stride
    fusion  -               -
    conv    256 × (1 × 1)   1 × 1
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1256, 256, kernel_size=1)
        self.act = nn.ReLU()

    def forward(self, img: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        """Combines img Tensor and feature vector into one 3D tensor, convolves over it.

        Assumes default image input size.

        Args:
            img (torch.Tensor): Output from Encoder
            feat_vec (torch.Tensor): Output from feature_extractor

        Returns:
            torch.Tensor: fusion of the img and feat_vec
        """
        # Reshape feat_vec to (batch, 1000, 1, 1)
        reshaped_feat_vec = torch.reshape(feat_vec, (*feat_vec.shape, 1, 1))
        # Repeat the vector (H/8 x W/8) times, so that (batch, 1000, H/8, W/8)
        repeated_feat_vec = reshaped_feat_vec.repeat(1, 1, *img.shape[-2:])
        # Swap the last two axes so that vec[:, :, x, y] == feat_vec for any x, y
        embs = torch.permute(repeated_feat_vec, (0, 1, 3, 2))
        fused = torch.cat((img, embs), 1)
        x = self.act(self.conv(fused))
        return x
