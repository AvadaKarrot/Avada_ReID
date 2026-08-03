from typing import Optional

import torch
from torch import nn

from ..outputs import BackboneOutput
from .base import BackboneAdapter
from .registry import register_backbone


@register_backbone("clip_vit_b16")
class CLIPViTB16Adapter(BackboneAdapter):
    """Adapter around the repository's CLIP visual encoder.

    Loading remains lazy so importing the new package does not download weights
    or require the legacy CLIP dependencies.
    """

    output_dim = 768

    def __init__(self, cfg=None, encoder: Optional[nn.Module] = None):
        super().__init__()
        if encoder is None:
            if cfg is None:
                raise ValueError("cfg is required when no CLIP encoder is supplied")
            try:
                from model.clip_loader import load_reid_clip
            except ImportError:
                from TransReID.model.clip_loader import load_reid_clip

            height, width = cfg.INPUT.SIZE_TRAIN
            stride = cfg.MODEL.STRIDE_SIZE[0]
            h_resolution = int((height - 16) // cfg.MODEL.STRIDE_SIZE[0] + 1)
            w_resolution = int((width - 16) // cfg.MODEL.STRIDE_SIZE[1] + 1)
            encoder = load_reid_clip(
                cfg, h_resolution, w_resolution, stride, trainer="CoOp"
            ).visual

        self.encoder = encoder

    def forward_features(self, images: torch.Tensor) -> BackboneOutput:
        outputs = self.encoder(images)
        if not isinstance(outputs, (tuple, list)) or len(outputs) < 3:
            raise RuntimeError(
                "The repository CLIP visual encoder must return token features"
            )

        # Legacy CLIP returns (last_tokens, tokens, projected_tokens).
        last_tokens, tokens, projected_tokens = outputs[:3]
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected [B, N, D] CLIP tokens, got {tokens.shape}")

        height = images.shape[-2] // 16
        width = images.shape[-1] // 16
        return BackboneOutput(
            global_feature=tokens[:, 0],
            patch_features=tokens[:, 1:],
            spatial_shape=(height, width),
            auxiliary_features={
                "last_global": last_tokens[:, 0],
                "projected_global": projected_tokens[:, 0],
            },
        )
