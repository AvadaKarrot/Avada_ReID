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
        id_logits = outputs.id_logits
        if id_logits is None and outputs.logits is not None:
            id_logits = (outputs.logits,)
        if not id_logits:
            raise RuntimeError("ReIDObjective requires model.train() logits")

        metric_features = outputs.metric_features or (outputs.raw_feature,)
        id_losses = [
            F.cross_entropy(
                logits,
                pids,
                label_smoothing=self.label_smoothing,
            )
            for logits in id_logits
        ]
        triplet_losses = [
            self.triplet(features, pids) for features in metric_features
        ]

        losses = {
            "id": sum(id_losses),
            "triplet": sum(triplet_losses),
        }

        total = (
            self.id_weight * losses["id"]
            + self.triplet_weight * losses["triplet"]
        )

        if self.caption_objective is not None and self.caption_weight > 0:
            alignment_feature = (
                outputs.alignment_feature
                if outputs.alignment_feature is not None
                else outputs.raw_feature
            )
            losses["caption"] = self.caption_objective(
                image_features=alignment_feature,
                captions=batch.get("captions"),
                valid_mask=batch.get("caption_mask"),
                pids=pids,
            )
            total = total + self.caption_weight * losses["caption"]

        losses["total"] = total
        return losses
