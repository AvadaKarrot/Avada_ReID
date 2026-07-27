import torch
from torch import nn
from torch.nn import functional as F


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss with safe handling of incomplete batches."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        distances = torch.cdist(features.float(), features.float(), p=2)
        same_identity = labels[:, None].eq(labels[None, :])
        same_identity.fill_diagonal_(False)
        different_identity = ~labels[:, None].eq(labels[None, :])

        valid = same_identity.any(dim=1) & different_identity.any(dim=1)
        if not valid.any():
            return features.sum() * 0.0

        hardest_positive = distances.masked_fill(
            ~same_identity, float("-inf")
        ).max(dim=1).values
        hardest_negative = distances.masked_fill(
            ~different_identity, float("inf")
        ).min(dim=1).values

        return F.margin_ranking_loss(
            hardest_negative[valid],
            hardest_positive[valid],
            torch.ones_like(hardest_negative[valid]),
            margin=self.margin,
        )
