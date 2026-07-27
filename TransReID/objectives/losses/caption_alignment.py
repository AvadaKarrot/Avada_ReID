from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class CaptionAlignmentObjective(nn.Module):
    """Optional source-only image/text alignment objective.

    ``text_encoder`` is injected so caption generation and tokenization remain
    independent of the visual ReID model.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        text_encoder: Callable[[Sequence[str]], torch.Tensor],
        projection_dim: int = 512,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_projection = nn.Linear(image_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)
        self.temperature = temperature

    def forward(
        self,
        image_features: torch.Tensor,
        captions: Optional[Sequence[str]],
        valid_mask: Optional[torch.Tensor],
        **_,
    ) -> torch.Tensor:
        if captions is None or valid_mask is None or not valid_mask.any():
            return image_features.sum() * 0.0

        valid_mask = valid_mask.to(
            device=image_features.device, dtype=torch.bool
        )
        selected_captions = [
            caption
            for caption, valid in zip(captions, valid_mask.cpu().tolist())
            if valid
        ]
        text_features = self.text_encoder(selected_captions)
        text_features = text_features.to(image_features.device)

        image_embeddings = F.normalize(
            self.image_projection(image_features[valid_mask]), dim=-1
        )
        text_embeddings = F.normalize(
            self.text_projection(text_features), dim=-1
        )
        logits = image_embeddings @ text_embeddings.t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)
        )
