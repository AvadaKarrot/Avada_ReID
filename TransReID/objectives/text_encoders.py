"""Text towers used only by source-domain training objectives."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


class LegacyCLIPTextEncoder(nn.Module):
    """Plain CLIP text tower loaded from the repository's CLIP checkpoint.

    This uses the same checkpoint family as the old Caption/MaPLe experiments,
    but deliberately builds the ordinary CoOp-compatible transformer because
    no learnable deep MaPLe prompts are passed by this adapter. MaPLe remains a
    separate model feature. Per-image captions never enter ``ReIDModel``.
    """

    def __init__(
        self,
        cfg=None,
        *,
        clip_model=None,
        tokenizer: Callable[..., torch.Tensor] | None = None,
        trainable: bool = False,
        truncate: bool = True,
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
                trainer="CoOp",
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
        self.truncate = bool(truncate)
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
        tokens = self.tokenizer(
            list(captions), truncate=self.truncate
        ).to(
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


class SigLIP2TextEncoder(nn.Module):
    """Frozen-by-default native SigLIP2 text tower.

    The returned native pooled text representation is the 768-D counterpart
    of the vision tower's pretrained attention-pooler output. Tokenization
    uses the fixed maximum-length padding contract from SigLIP2 pretraining.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        text_model=None,
        tokenizer=None,
        trainable: bool = False,
    ):
        super().__init__()
        if text_model is None or tokenizer is None:
            if not model_name:
                raise ValueError(
                    "model_name is required unless both text_model and "
                    "tokenizer are injected"
                )
            try:
                from transformers import Siglip2TextModel, Siglip2Tokenizer
            except ImportError as exc:
                raise ImportError(
                    "SigLIP2 text alignment requires transformers"
                ) from exc
            if text_model is None:
                text_model = Siglip2TextModel.from_pretrained(model_name)
            if tokenizer is None:
                # Use the explicit class so the training contract does not
                # depend on tokenizer_class metadata from a Hub revision.
                tokenizer = Siglip2Tokenizer.from_pretrained(model_name)

        self.text_model = text_model
        self.tokenizer = tokenizer
        self.output_dim = int(
            getattr(text_model.config, "hidden_size", 768)
        )
        self.max_length = int(
            getattr(text_model.config, "max_position_embeddings", 64)
        )
        self._trainable = bool(trainable)
        if not self._trainable:
            self.requires_grad_(False)
            super().train(False)

    def train(self, mode: bool = True):
        return super().train(mode if self._trainable else False)

    def forward(self, captions: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.text_model.parameters()).device
        inputs = {name: value.to(device) for name, value in inputs.items()}
        outputs = self.text_model(**inputs, return_dict=True)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            raise RuntimeError("SigLIP2 text tower returned no pooler_output")
        return pooled
