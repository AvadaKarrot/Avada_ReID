from typing import Optional

import torch
from torch import nn

from ..outputs import BackboneOutput
from .base import BackboneAdapter
from .registry import register_backbone


@register_backbone("dinov3_vit_b16")
class DINOv3Adapter(BackboneAdapter):
    """Image-only DINOv3 adapter that excludes register tokens locally."""

    output_dim = 768

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        encoder: Optional[nn.Module] = None,
        register_tokens: int = 4,
    ):
        super().__init__()
        if encoder is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "DINOv3 requires transformers. Install the project environment "
                    "before constructing this backbone."
                ) from exc
            encoder = AutoModel.from_pretrained(model_name)

        self.encoder = encoder
        self.register_tokens = register_tokens
        self.patch_size = int(getattr(encoder.config, "patch_size", 16))
        self.output_dim = int(getattr(encoder.config, "hidden_size", self.output_dim))

    def forward_features(self, images: torch.Tensor) -> BackboneOutput:
        outputs = self.encoder(
            pixel_values=images,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state
        patch_start = 1 + self.register_tokens

        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        patch_features = tokens[:, patch_start:]
        expected_patches = height * width
        if patch_features.shape[1] != expected_patches:
            raise RuntimeError(
                "Unexpected DINOv3 token layout: "
                f"expected {expected_patches} patches, got {patch_features.shape[1]}"
            )

        return BackboneOutput(
            global_feature=tokens[:, 0],
            patch_features=patch_features,
            spatial_shape=(height, width),
        )
