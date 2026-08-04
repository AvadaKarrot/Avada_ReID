from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

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
        static_position_embedding: bool = False,
        target_image_size=None,
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
        self.static_position_embedding = bool(static_position_embedding)
        self.position_grid = None
        if self.static_position_embedding:
            if target_image_size is None:
                raise ValueError(
                    "target_image_size is required for a static position embedding"
                )
            target_height, target_width = map(int, target_image_size)
            if target_height % self.patch_size or target_width % self.patch_size:
                raise ValueError(
                    "Static SigLIP2 image size must be divisible by patch_size"
                )
            self.position_grid = (
                target_height // self.patch_size,
                target_width // self.patch_size,
            )
            self._resize_position_embedding_once(self.position_grid)

    def _resize_position_embedding_once(self, target_grid):
        vision_tower = getattr(self.encoder, "vision_model", self.encoder)
        embeddings = getattr(vision_tower, "embeddings", None)
        position_embedding = getattr(embeddings, "position_embedding", None)
        if not isinstance(position_embedding, nn.Embedding):
            raise RuntimeError(
                "SigLIP2 static position mode requires an nn.Embedding table"
            )

        weight = position_embedding.weight.detach()
        token_count, hidden_size = weight.shape
        old_size = int(token_count ** 0.5)
        if old_size * old_size != token_count:
            raise RuntimeError(
                "Pretrained SigLIP2 position table must form a square grid"
            )
        resized = F.interpolate(
            weight.reshape(1, old_size, old_size, hidden_size).permute(0, 3, 1, 2),
            size=target_grid,
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(-1, hidden_size)

        replacement = nn.Embedding(resized.shape[0], hidden_size).to(
            device=weight.device, dtype=weight.dtype
        )
        with torch.no_grad():
            replacement.weight.copy_(resized)
        embeddings.position_embedding = replacement
        embeddings.position_ids = torch.arange(
            resized.shape[0], device=weight.device
        ).expand((1, -1))

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
        if self.position_grid is not None and spatial_shape != self.position_grid:
            raise ValueError(
                "Static SigLIP2 position grid does not match the input: "
                f"expected {self.position_grid}, got {spatial_shape}"
            )

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
                interpolate_pos_encoding=not self.static_position_embedding,
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
