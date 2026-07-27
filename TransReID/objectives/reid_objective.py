from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .losses import BatchHardTripletLoss


class ReIDObjective(nn.Module):
    """Single configurable objective for every visual backbone."""

    def __init__(
        self,
        triplet_margin: float = 0.3,
        id_weight: float = 1.0,
        triplet_weight: float = 1.0,
        label_smoothing: float = 0.0,
        caption_objective: Optional[nn.Module] = None,
        caption_weight: float = 0.0,
    ):
        super().__init__()
        self.triplet = BatchHardTripletLoss(margin=triplet_margin)
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight
        self.label_smoothing = label_smoothing
        self.caption_objective = caption_objective
        self.caption_weight = caption_weight

    def forward(self, outputs, batch) -> Dict[str, torch.Tensor]:
        pids = batch["pids"].to(outputs.raw_feature.device)
        if outputs.logits is None:
            raise RuntimeError("ReIDObjective requires model.train() logits")

        losses = {
            "id": F.cross_entropy(
                outputs.logits,
                pids,
                label_smoothing=self.label_smoothing,
            ),
            "triplet": self.triplet(outputs.raw_feature, pids),
        }

        total = (
            self.id_weight * losses["id"]
            + self.triplet_weight * losses["triplet"]
        )

        if self.caption_objective is not None and self.caption_weight > 0:
            losses["caption"] = self.caption_objective(
                image_features=outputs.raw_feature,
                captions=batch.get("captions"),
                valid_mask=batch.get("caption_mask"),
                pids=pids,
            )
            total = total + self.caption_weight * losses["caption"]

        losses["total"] = total
        return losses
