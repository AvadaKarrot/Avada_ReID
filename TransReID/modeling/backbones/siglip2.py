import copy
from typing import Mapping, Optional

import torch
from torch import nn
from torch.nn import functional as F

from ..outputs import BackboneOutput
from .base import BackboneAdapter
from .registry import register_backbone


def resize_siglip2_position_embedding(
    position_embedding: torch.Tensor,
    target_grid,
    source_grid=None,
) -> torch.Tensor:
    """Linearly resize a SigLIP2 patch-position table to a rectangular grid.

    Unlike CLIP ViT, SigLIP2's position table contains patch positions only;
    there is no leading CLS position to split off. ``target_grid`` and the
    optional ``source_grid`` are ``(height, width)`` in patch units. The
    function accepts either ``[num_patches, hidden_dim]`` or
    ``[1, num_patches, hidden_dim]`` and preserves that input rank.

    SigLIP2's reference high-resolution fine-tuning implementation uses
    ``scipy.ndimage.zoom(order=1)``. PyTorch bilinear interpolation is the
    differentiable first-order equivalent and also supports a non-square
    target such as the 16x8 grid produced by 256x128 ReID inputs.
    """

    if position_embedding.ndim not in (2, 3):
        raise ValueError(
            "SigLIP2 position embedding must be [N, D] or [1, N, D], "
            f"got {tuple(position_embedding.shape)}"
        )
    if position_embedding.ndim == 3:
        if position_embedding.shape[0] != 1:
            raise ValueError(
                "Batched SigLIP2 position embedding must have leading size 1"
            )
        table = position_embedding[0]
    else:
        table = position_embedding

    target_height, target_width = map(int, target_grid)
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target_grid dimensions must be positive")

    token_count, hidden_size = table.shape
    if source_grid is None:
        source_size = int(token_count ** 0.5)
        if source_size * source_size != token_count:
            raise ValueError(
                "A non-square source position table requires source_grid"
            )
        source_height = source_width = source_size
    else:
        source_height, source_width = map(int, source_grid)
        if source_height * source_width != token_count:
            raise ValueError(
                "source_grid does not match the position-table token count"
            )

    if (source_height, source_width) == (target_height, target_width):
        return position_embedding

    resized = F.interpolate(
        table.reshape(
            1, source_height, source_width, hidden_size
        ).permute(0, 3, 1, 2),
        size=(target_height, target_width),
        # Match SigLIP2's scipy.ndimage.zoom(order=1) contract.
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).reshape(-1, hidden_size)
    return resized.unsqueeze(0) if position_embedding.ndim == 3 else resized


class MAPHead(nn.Module):
    """PyTorch equivalent of big_vision's multihead attention pooler.

    A learned probe attends to all patch tokens, then a pre-normalized MLP
    residual refines the single pooled token.  Released Hugging Face SigLIP2
    checkpoints already provide one pretrained instance after the final
    encoder layer. This implementation remains a compatibility fallback for
    encoders that do not expose that native head.
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

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                f"MAPHead expects tokens shaped [B, N, D], got {tokens.shape}"
            )
        if tokens.shape[-1] != self.hidden_size:
            raise ValueError(
                "MAPHead token dimension does not match hidden_size: "
                f"got {tokens.shape[-1]}, expected {self.hidden_size}"
            )
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    "MAPHead attention mask must match [B, N]: "
                    f"got {tuple(attention_mask.shape)}, expected "
                    f"{tuple(tokens.shape[:2])}"
                )
            key_padding_mask = ~attention_mask.to(dtype=torch.bool)
        probe = self.probe.expand(tokens.shape[0], -1, -1)
        pooled = self.attention(
            probe,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
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
            native_map_head = getattr(vision_tower, "head", None)
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
            if isinstance(native_map_head, nn.Module):
                # Block 11 has no dedicated pooler in the checkpoint. Give
                # its independent ReID branch a pretrained initialization by
                # cloning SigLIP2's final MAP head instead of starting random.
                self.penultimate_map_head = copy.deepcopy(native_map_head)
            else:
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
        hidden_size = weight.shape[1]
        try:
            resized = resize_siglip2_position_embedding(weight, target_grid)
        except ValueError as error:
            raise RuntimeError(str(error)) from error

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


def _masked_token_mean(tokens, attention_mask):
    if attention_mask.shape != tokens.shape[:2]:
        raise ValueError(
            "Patch attention mask must match token dimensions: "
            f"got {tuple(attention_mask.shape)}, expected "
            f"{tuple(tokens.shape[:2])}"
        )
    weights = attention_mask.to(device=tokens.device, dtype=tokens.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1) / denominator


@register_backbone("siglip2_base_patch16_naflex")
class SigLIP2NaFlexAdapter(BackboneAdapter):
    """Official SigLIP2 NaFlex visual contract with mask-aware pooling."""

    output_dim = 768
    secondary_dim = 768

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-naflex",
        encoder: Optional[nn.Module] = None,
        penultimate_map_pooler: bool = False,
    ):
        super().__init__()
        if encoder is None:
            try:
                from transformers import Siglip2VisionModel
            except ImportError as exc:
                raise ImportError(
                    "SigLIP2 NaFlex requires a Transformers release with "
                    "Siglip2VisionModel"
                ) from exc
            encoder = Siglip2VisionModel.from_pretrained(model_name)

        self.encoder = encoder
        self.output_dim = int(
            getattr(encoder.config, "hidden_size", self.output_dim)
        )
        self.secondary_dim = self.output_dim
        self.penultimate_map_pooler_enabled = bool(penultimate_map_pooler)
        self.penultimate_post_layernorm = None
        self.penultimate_map_head = None
        if self.penultimate_map_pooler_enabled:
            vision_tower = getattr(self.encoder, "vision_model", self.encoder)
            post_layernorm = getattr(vision_tower, "post_layernorm", None)
            native_map_head = getattr(vision_tower, "head", None)
            if not isinstance(native_map_head, nn.Module):
                raise RuntimeError(
                    "NaFlex penultimate MAP requires the checkpoint's native "
                    "mask-aware MAP head"
                )
            if isinstance(post_layernorm, nn.Module):
                self.penultimate_post_layernorm = copy.deepcopy(post_layernorm)
            else:
                self.penultimate_post_layernorm = nn.LayerNorm(
                    self.output_dim,
                    eps=float(getattr(encoder.config, "layer_norm_eps", 1e-6)),
                )
            self.penultimate_map_head = copy.deepcopy(native_map_head)

    @staticmethod
    def _validate_inputs(image_inputs):
        if not isinstance(image_inputs, Mapping):
            raise TypeError(
                "SigLIP2 NaFlex expects a mapping containing pixel_values, "
                "pixel_attention_mask and spatial_shapes"
            )
        required = {
            "pixel_values",
            "pixel_attention_mask",
            "spatial_shapes",
        }
        missing = required.difference(image_inputs)
        if missing:
            raise KeyError(
                f"NaFlex visual inputs are missing {sorted(missing)}"
            )
        pixel_values = image_inputs["pixel_values"]
        attention_mask = image_inputs["pixel_attention_mask"]
        spatial_shapes = image_inputs["spatial_shapes"]
        if pixel_values.ndim != 3:
            raise ValueError(
                "NaFlex pixel_values must be [B, N, patch_dim], got "
                f"{tuple(pixel_values.shape)}"
            )
        if attention_mask.shape != pixel_values.shape[:2]:
            raise ValueError(
                "NaFlex pixel_attention_mask must match [B, N]"
            )
        if spatial_shapes.shape != (pixel_values.shape[0], 2):
            raise ValueError(
                "NaFlex spatial_shapes must be [B, 2], got "
                f"{tuple(spatial_shapes.shape)}"
            )
        return pixel_values, attention_mask, spatial_shapes

    def forward_features(self, image_inputs: Mapping[str, torch.Tensor]):
        pixel_values, attention_mask, spatial_shapes = self._validate_inputs(
            image_inputs
        )
        attention_mask = attention_mask.to(dtype=torch.bool)

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
                pixel_values=pixel_values,
                pixel_attention_mask=attention_mask,
                spatial_shapes=spatial_shapes,
                output_hidden_states=self.penultimate_map_pooler_enabled,
            )
        finally:
            if hook is not None:
                hook.remove()

        tokens = outputs.last_hidden_state
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise RuntimeError(
                "SigLIP2 NaFlex checkpoint returned no native MAP output"
            )
        pre_norm_tokens = pre_norm[0] if pre_norm else tokens
        auxiliary_features = {
            "alignment_global": pooled,
            "patch_attention_mask": attention_mask,
            "spatial_shapes": spatial_shapes,
        }
        if self.penultimate_map_pooler_enabled:
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None or len(hidden_states) < 3:
                raise RuntimeError(
                    "NaFlex encoder did not return penultimate hidden states"
                )
            penultimate_tokens = self.penultimate_post_layernorm(
                hidden_states[-2]
            )
            auxiliary_features["penultimate_map_global"] = (
                self.penultimate_map_head(
                    penultimate_tokens,
                    attention_mask,
                )
            )

        return BackboneOutput(
            global_feature=_masked_token_mean(tokens, attention_mask),
            patch_features=tokens,
            spatial_shape=None,
            pre_norm_global=_masked_token_mean(
                pre_norm_tokens, attention_mask
            ),
            secondary_global=pooled,
            auxiliary_features=auxiliary_features,
        )
