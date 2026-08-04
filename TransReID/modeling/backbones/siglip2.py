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
    secondary_dim = 768

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        if encoder is None:
            try:
                from transformers import SiglipVisionModel
            except ImportError as exc:
                raise ImportError(
                    "SigLIP2 requires transformers. Install the project environment "
                    "before constructing this backbone."
                ) from exc
            # The released SigLIP2 Base checkpoint is stored with the
            # ``siglip`` configuration/API contract.  Loading it through
            # Siglip2VisionModel changes the patch-embedding contract and
            # silently reinitializes incompatible weights on recent
            # Transformers versions.
            encoder = SiglipVisionModel.from_pretrained(model_name)

        self.encoder = encoder
        self.patch_size = int(getattr(encoder.config, "patch_size", 16))
        self.output_dim = int(getattr(encoder.config, "hidden_size", self.output_dim))
        self.secondary_dim = self.output_dim

    def forward_features(self, images: torch.Tensor) -> BackboneOutput:
        if images.ndim != 4:
            raise ValueError(
                f"Expected SigLIP2 images shaped [B, C, H, W], got {images.shape}"
            )
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                "SigLIP2 image height and width must be divisible by patch_size: "
                f"got {(height, width)} and patch_size={self.patch_size}"
            )
        spatial_shape = (height // self.patch_size, width // self.patch_size)

        pre_norm = []
        vision_tower = getattr(self.encoder, "vision_model", self.encoder)
        post_layernorm = getattr(vision_tower, "post_layernorm", None)
        hook = None
        if post_layernorm is not None:
            hook = post_layernorm.register_forward_pre_hook(
                lambda _module, inputs: pre_norm.append(inputs[0])
            )
        try:
            outputs = self.encoder(
                pixel_values=images,
                interpolate_pos_encoding=True,
                return_dict=True,
            )
        finally:
            if hook is not None:
                hook.remove()
        tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = tokens.mean(dim=1)
        pre_norm_tokens = pre_norm[0] if pre_norm else tokens

        return BackboneOutput(
            global_feature=tokens.mean(dim=1),
            patch_features=tokens,
            spatial_shape=spatial_shape,
            pre_norm_global=pre_norm_tokens.mean(dim=1),
            secondary_global=pooled,
            auxiliary_features={"alignment_global": pooled},
        )
