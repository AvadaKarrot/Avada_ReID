"""Text towers used only by source-domain training objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


class LegacyCLIPTextEncoder(nn.Module):
    """CLIP text tower loaded through the legacy MaPLe-capable loader.

    The loader supplies the same CLIP checkpoint family used by the old
    Caption/MaPLe experiments. This adapter does not claim to implement MaPLe:
    MaPLe's learnable deep visual/text prompts are a separate model feature.
    Only the plain text tower is retained here, and per-image captions never
    enter ``ReIDModel``.
    """

    def __init__(
        self,
        cfg=None,
        *,
        clip_model=None,
        tokenizer: Callable[[Sequence[str]], torch.Tensor] | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        if clip_model is None:
            if cfg is None:
                raise ValueError("cfg is required when clip_model is omitted")
            from model.clip_loader import load_reid_clip
            from model.maple.clip import clip

            height, width = cfg.INPUT.SIZE_TRAIN
            patch_size = 16
            stride_h, stride_w = cfg.MODEL.STRIDE_SIZE
            h_resolution = int((height - patch_size) // stride_h + 1)
            w_resolution = int((width - patch_size) // stride_w + 1)
            clip_model = load_reid_clip(
                cfg,
                h_resolution,
                w_resolution,
                stride_h,
                trainer="MaPLe",
            )
            tokenizer = clip.tokenize
        if tokenizer is None:
            raise ValueError("tokenizer is required with an injected CLIP model")

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.tokenizer = tokenizer
        self.output_dim = int(self.text_projection.shape[-1])
        self._trainable = bool(trainable)

        if not trainable:
            self.requires_grad_(False)
            super().train(False)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def train(self, mode: bool = True):
        return super().train(mode if self._trainable else False)

    def forward(self, captions: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(captions)).to(
            self.token_embedding.weight.device
        )
        x = self.token_embedding(tokens).to(self.dtype)
        x = x + self.positional_embedding.to(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).to(self.dtype)
        row_ids = torch.arange(tokens.shape[0], device=tokens.device)
        return x[row_ids, tokens.argmax(dim=-1)] @ self.text_projection
