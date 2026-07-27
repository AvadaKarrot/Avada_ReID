from typing import Optional

import torch
from torch import nn

from ..outputs import BackboneOutput
from .base import BackboneAdapter
from .registry import register_backbone


@register_backbone("siglip2_base_patch16")
class SigLIP2Adapter(BackboneAdapter):
    """SigLIP2 visual-tower adapter with no text dependency at inference."""

    output_dim = 768

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        if encoder is None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "SigLIP2 requires transformers. Install the project environment "
                    "before constructing this backbone."
                ) from exc
            full_model = AutoModel.from_pretrained(model_name)
            encoder = getattr(full_model, "vision_model", full_model)

        self.encoder = encoder
        self.patch_size = int(getattr(encoder.config, "patch_size", 16))
        self.output_dim = int(getattr(encoder.config, "hidden_size", self.output_dim))

    def forward_features(self, images: torch.Tensor) -> BackboneOutput:
        outputs = self.encoder(
            pixel_values=images,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = tokens.mean(dim=1)

        height = images.shape[-2] // self.patch_size
        width = images.shape[-1] // self.patch_size
        return BackboneOutput(
            global_feature=pooled,
            patch_features=tokens,
            spatial_shape=(height, width),
        )
