from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch


@dataclass
class BackboneOutput:
    """Backbone features with an explicit global/local contract."""

    global_feature: torch.Tensor
    patch_features: Optional[torch.Tensor] = None
    spatial_shape: Optional[Tuple[int, int]] = None
    # Global representation before the backbone's final normalization.  This
    # is the first metric branch in the verified CLIP-ReID recipe.
    pre_norm_global: Optional[torch.Tensor] = None
    # A complementary native global representation.  CLIP supplies its
    # pretrained visual projection, DINOv3 uses mean patch features, and
    # SigLIP2 uses its pretrained attention pooler.
    secondary_global: Optional[torch.Tensor] = None
    auxiliary_features: Optional[Mapping[str, torch.Tensor]] = None


@dataclass
class ReIDOutput:
    """Stable output consumed by objectives and evaluators."""

    embedding: torch.Tensor
    raw_feature: torch.Tensor
    logits: Optional[torch.Tensor] = None
    patch_features: Optional[torch.Tensor] = None
    id_logits: Optional[Tuple[torch.Tensor, ...]] = None
    metric_features: Optional[Tuple[torch.Tensor, ...]] = None
    alignment_feature: Optional[torch.Tensor] = None
