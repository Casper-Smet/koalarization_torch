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
from torch import nn
from timm import create_model


from .decoder import Decoder
from .encoder import Encoder


class Colorization(nn.Module):
    """Colorization architecture with Encoder, Feature extractor, and Decoder."""
    def __init__(self, feature_extractor: Any = None) -> None:
        super().__init__()
        self.encode = Encoder()
        self.decode = Decoder()
        if feature_extractor is None:
            # lazy import for typing
            from timm.models.inception_resnet_v2 import InceptionResnetV2

            self.feat_extr: InceptionResnetV2 = create_model(
                "inception_resnet_v2", pretrained=True
            )
        else:
            self.feat_extr = feature_extractor


def test() -> None:
    """Test function for setup tools."""
    print("Test function")
