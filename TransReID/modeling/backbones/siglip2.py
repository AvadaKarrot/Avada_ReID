import copy
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..outputs import BackboneOutput
from .base import BackboneAdapter
from .registry import register_backbone


class MAPHead(nn.Module):
    """PyTorch equivalent of big_vision's multihead attention pooler.

    A learned probe attends to all patch tokens, then a pre-normalized MLP
    residual refines the single pooled token.  Released Hugging Face SigLIP2
    checkpoints already provide one pretrained instance after the final
    encoder layer.  This implementation supplies an independent trainable
    instance for the penultimate-layer ReID feature.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 12,
        mlp_dim: Optional[int] = None,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                "MAPHead hidden_size must be divisible by num_heads: "
                f"got hidden_size={hidden_size}, num_heads={num_heads}"
            )
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim or 4 * hidden_size)

        self.probe = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, hidden_size),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.probe)
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        if self.attention.in_proj_bias is not None:
            nn.init.zeros_(self.attention.in_proj_bias)
        if self.attention.out_proj.bias is not None:
            nn.init.zeros_(self.attention.out_proj.bias)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"MAPHead expects tokens shaped [B, N, D], got {tokens.shape}"
            )
        if tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                "MAPHead token dimension does not match hidden_size: "
                f"got {tokens.shape[-1]}, expected {self.hidden_size}"
            )
        probe = self.probe.expand(tokens.shape[0], -1, -1)
        pooled = self.attention(probe, tokens, tokens, need_weights=False)[0]
        pooled = pooled + self.mlp(self.layernorm(pooled))
        return pooled[:, 0]


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
        penultimate_map_pooler: bool = False,
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
        self.penultimate_map_pooler_enabled = bool(
            penultimate_map_pooler
        )
        self.penultimate_post_layernorm = None
        self.penultimate_map_head = None
        if self.penultimate_map_pooler_enabled:
            vision_tower = getattr(self.encoder, "vision_model", self.encoder)
            post_layernorm = getattr(vision_tower, "post_layernorm", None)
            layer_norm_eps = float(
                getattr(encoder.config, "layer_norm_eps", 1e-6)
            )
            if isinstance(post_layernorm, nn.Module):
                # Match the final native path structurally, but do not share
                # normalization parameters between blocks 11 and 12.
                self.penultimate_post_layernorm = copy.deepcopy(
                    post_layernorm
                )
            else:
                self.penultimate_post_layernorm = nn.LayerNorm(
                    self.output_dim, eps=layer_norm_eps
                )
            self.penultimate_map_head = MAPHead(
                hidden_size=self.output_dim,
                num_heads=int(
                    getattr(encoder.config, "num_attention_heads", 12)
                ),
                mlp_dim=getattr(encoder.config, "intermediate_size", None),
                layer_norm_eps=layer_norm_eps,
            )
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
                output_hidden_states=self.penultimate_map_pooler_enabled,
                return_dict=True,
            )
        finally:
            if hook is not None:
                hook.remove()
        tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise RuntimeError(
                "SigLIP2 encoder must provide the pretrained final-layer "
                "MAPHead output as pooler_output"
            )
        pre_norm_tokens = pre_norm[0] if pre_norm else tokens
        auxiliary_features = {"alignment_global": pooled}
        if self.penultimate_map_pooler_enabled:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None or len(hidden_states) < 3:
                raise RuntimeError(
                    "SigLIP2 encoder must return embedding and per-layer "
                    "hidden states so the penultimate layer can be pooled"
                )
            # Hugging Face returns embeddings, block 1, ..., block 11,
            # block 12. Therefore -2 is block 11 for the 12-layer Base model.
            penultimate_tokens = self.penultimate_post_layernorm(
                hidden_states[-2]
            )
            auxiliary_features["penultimate_map_global"] = (
                self.penultimate_map_head(penultimate_tokens)
            )

        return BackboneOutput(
            global_feature=tokens.mean(dim=1),
            patch_features=tokens,
            spatial_shape=spatial_shape,
            pre_norm_global=pre_norm_tokens.mean(dim=1),
            secondary_global=pooled,
            auxiliary_features=auxiliary_features,
        )
