from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from objectives.distributed import (
    distributed_ready,
    gather_variable,
    gather_variable_with_grad,
    globally_normalized_local_sum,
)


class CaptionAlignmentObjective(nn.Module):
    """PID-aware source-only image/text alignment objective.

    ``text_encoder`` is injected so caption generation and tokenization remain
    independent of the visual ReID model. Samples with the same PID are
    multi-positive pairs instead of false negatives.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        text_encoder: Callable[[Sequence[str]], torch.Tensor],
        projection_dim: int = 512,
        temperature: float = 0.07,
        positive_mode: str = "pid",
        use_projection: bool = True,
        gather_across_ranks: bool = False,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.use_projection = bool(use_projection)
        if self.use_projection:
            self.image_projection = nn.Linear(
                image_dim, projection_dim, bias=False
            )
            self.text_projection = nn.Linear(
                text_dim, projection_dim, bias=False
            )
        else:
            if image_dim != text_dim:
                raise ValueError(
                    "Projection-free alignment requires equal image and "
                    "text dimensions"
                )
            self.image_projection = nn.Identity()
            self.text_projection = nn.Identity()
        self.temperature = temperature
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.positive_mode = str(positive_mode).lower()
        self.gather_across_ranks = bool(gather_across_ranks)
        self.last_metrics = {}
        if self.positive_mode not in {"pid", "instance"}:
            raise ValueError(
                "positive_mode must be either 'pid' or 'instance'"
            )

    def frozen_text_encoder_state_is_reconstructible(self) -> bool:
        """Return whether text weights may be restored from pretrained files.

        Frozen text towers are immutable inputs to the alignment objective and
        are loaded before a training checkpoint is restored.  Persisting them
        in every checkpoint only duplicates the pretrained model.  A future
        trainable text tower is deliberately kept in the checkpoint.
        """

        if not isinstance(self.text_encoder, nn.Module):
            return False
        parameters = tuple(self.text_encoder.parameters())
        return bool(parameters) and all(
            not parameter.requires_grad for parameter in parameters
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        if self.frozen_text_encoder_state_is_reconstructible():
            text_prefix = f"{prefix}text_encoder."
            for key in tuple(state):
                if key.startswith(text_prefix):
                    del state[key]
        return state

    @staticmethod
    def _positive_mask(
        pids: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "pid":
            return pids[:, None].eq(pids[None, :])
        return torch.eye(
            pids.shape[0],
            dtype=torch.bool,
            device=pids.device,
        )

    @staticmethod
    def _multi_positive_nce(
        logits: torch.Tensor,
        positive_mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if not positive_mask.any(dim=1).all():
            raise RuntimeError("Every contrastive anchor needs a positive")
        positive_logits = logits.masked_fill(
            ~positive_mask, torch.finfo(logits.dtype).min
        )
        values = (
            torch.logsumexp(logits, dim=1)
            - torch.logsumexp(positive_logits, dim=1)
        )
        if reduction == "none":
            return values
        if reduction != "mean":
            raise ValueError("reduction must be 'mean' or 'none'")
        return values.mean()

    def forward(
        self,
        image_features: torch.Tensor,
        captions: Optional[Sequence[str]],
        valid_mask: Optional[torch.Tensor],
        pids: torch.Tensor,
        **_,
    ) -> torch.Tensor:
        self.last_metrics = {}
        use_global_candidates = (
            self.gather_across_ranks and distributed_ready()
        )
        if (
            not use_global_candidates
            and (
                captions is None
                or valid_mask is None
                or not valid_mask.any()
            )
        ):
            return image_features.sum() * 0.0

        if captions is None or valid_mask is None:
            valid_mask = torch.zeros(
                image_features.shape[0],
                device=image_features.device,
                dtype=torch.bool,
            )
            captions = tuple("" for _ in range(image_features.shape[0]))

        valid_mask = valid_mask.to(
            device=image_features.device, dtype=torch.bool
        )
        if valid_mask.numel() != image_features.shape[0]:
            raise ValueError("caption_mask and image batch sizes differ")
        if pids.shape[0] != image_features.shape[0]:
            raise ValueError("PID and image batch sizes differ")
        if len(captions) != image_features.shape[0]:
            raise ValueError("Caption and image batch sizes differ")
        selected_captions = [
            caption
            for caption, valid in zip(captions, valid_mask.cpu().tolist())
            if valid
        ]
        if selected_captions:
            text_features = self.text_encoder(selected_captions)
            text_features = text_features.to(image_features.device)
        else:
            text_dim = (
                self.text_projection.in_features
                if isinstance(self.text_projection, nn.Linear)
                else image_features.shape[-1]
            )
            text_features = image_features.new_empty((0, text_dim))
        if text_features.shape[0] != len(selected_captions):
            raise ValueError(
                "Text encoder output and selected Caption counts differ"
            )
        selected_pids = pids[valid_mask]

        image_embeddings = F.normalize(
            self.image_projection(image_features[valid_mask]), dim=-1
        )
        text_embeddings = F.normalize(
            self.text_projection(text_features), dim=-1
        )
        if not use_global_candidates:
            logits = image_embeddings @ text_embeddings.t() / self.temperature
            self.last_metrics = {
                "caption_local_anchors": image_embeddings.new_tensor(
                    float(image_embeddings.shape[0])
                ),
                "caption_global_candidates": image_embeddings.new_tensor(
                    float(text_embeddings.shape[0])
                ),
                "caption_rank_gather": image_embeddings.new_tensor(0.0),
            }
            positive_mask = self._positive_mask(
                selected_pids,
                self.positive_mode,
            )
            return 0.5 * (
                self._multi_positive_nce(logits, positive_mask)
                + self._multi_positive_nce(logits.t(), positive_mask.t())
            )

        global_images, layout = gather_variable_with_grad(image_embeddings)
        global_texts, text_layout = gather_variable_with_grad(text_embeddings)
        if layout != text_layout:
            raise RuntimeError("Image/text distributed gather layouts differ")
        global_pids, _ = gather_variable(selected_pids, layout)
        self.last_metrics = {
            "caption_local_anchors": image_embeddings.new_tensor(
                float(image_embeddings.shape[0])
            ),
            "caption_global_candidates": image_embeddings.new_tensor(
                float(layout.global_size)
            ),
            "caption_rank_gather": image_embeddings.new_tensor(1.0),
        }
        if layout.global_size == 0:
            return image_features.sum() * 0.0

        image_logits = (
            image_embeddings @ global_texts.t() / self.temperature
        )
        text_logits = text_embeddings @ global_images.t() / self.temperature
        if self.positive_mode == "pid":
            positive_mask = selected_pids[:, None].eq(global_pids[None, :])
        else:
            positive_mask = torch.zeros(
                image_embeddings.shape[0],
                layout.global_size,
                dtype=torch.bool,
                device=image_embeddings.device,
            )
            local_indices = torch.arange(
                image_embeddings.shape[0], device=image_embeddings.device
            )
            positive_mask[
                local_indices, layout.local_offset + local_indices
            ] = True

        image_values = self._multi_positive_nce(
            image_logits, positive_mask, reduction="none"
        )
        text_values = self._multi_positive_nce(
            text_logits, positive_mask, reduction="none"
        )
        return 0.5 * (
            globally_normalized_local_sum(image_values)
            + globally_normalized_local_sum(text_values)
        )
