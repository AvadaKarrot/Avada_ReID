from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class BackboneOutput:
    """Backbone features with an explicit global/local contract."""

    global_feature: torch.Tensor
    patch_features: Optional[torch.Tensor] = None
    spatial_shape: Optional[Tuple[int, int]] = None


@dataclass
class ReIDOutput:
    """Stable output consumed by objectives and evaluators."""

    embedding: torch.Tensor
    raw_feature: torch.Tensor
    logits: Optional[torch.Tensor] = None
    patch_features: Optional[torch.Tensor] = None
