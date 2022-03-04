"""Network

Module containing Koalarization network architecture.

The original deep-koalarization paper mentions that the feature extraction branch
has the same architecture as the Inception-Resnet-v2 model, excluding *the last softmax layer*.
As far as I can tell, the timm version of Inception-Resnet-v2 does not include this softmax layer.

Colori*z*ation vs Colori*s*ation:
To stay with the naming scheme in the original paper, we use a 'z'.
"""

from typing import Any

import torch
from timm import create_model
from torch import nn

try:
    # There's a breaking change in the newest version of PILL that torchvision relies on
    from torchvision.transforms import Resize
except ImportError as ie:

    if "cannot import name 'PILLOW_VERSION'" in ie.msg:
        import PIL

        PIL.PILLOW_VERSION = PIL.__version__

        from torchvision.transforms import Resize
    else:
        raise ie

from .decoder import Decoder
from .encoder import Encoder
from .fusion import Fusion


class Colorization(nn.Module):
    """Colorization architecture with Encoder, Feature extractor, and Decoder."""

    def __init__(self, feature_extractor: Any = None) -> None:
        super().__init__()
        # Data preprocessing for the encoder
        self.resize = Resize((224, 224))
        # Encoder branch
        self.encode = Encoder()
        # Feature extractor branch
        if feature_extractor is None:
            # lazy import for typing
            from timm.models.inception_resnet_v2 import InceptionResnetV2

            self.feat_extr: InceptionResnetV2 = create_model(
                "inception_resnet_v2", pretrained=True
            )
        else:
            self.feat_extr = feature_extractor
        # Feature extractor is trained seperately
        self.feat_extr.requires_grad_(False)
        # Fusion layer
        self.fusion = Fusion()
        self.decode = Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract high level features from image
        feat_vec: torch.Tensor = self.feat_extr(x)
        # Take one colour channel, and resize
        single_channel_enc: torch.Tensor = self.resize(x[:, [0]])
        # Extract mid level features from image
        encoded_img: torch.Tensor = self.encode(single_channel_enc)
        # Fuze high and mid level features
        x = self.fusion(encoded_img, feat_vec)
        # Decode fuzed features
        x = self.decode(x)
        return x


def test() -> None:
    """Test function for setup tools."""
    print("Test function")
