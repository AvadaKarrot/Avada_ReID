import torch
from torch import nn
from torch.nn import functional as F

from objectives.distributed import (
    distributed_ready,
    gather_variable,
    gather_variable_with_grad,
    globally_normalized_local_sum,
)


class BatchHardTripletLoss(nn.Module):
    """Batch-hard triplet loss with safe handling of incomplete batches."""

    def __init__(
        self,
        margin: float = 0.3,
        gather_across_ranks: bool = False,
    ):
        super().__init__()
        self.margin = margin
        self.gather_across_ranks = bool(gather_across_ranks)
        self.last_metrics = {}

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        use_global_candidates = (
            self.gather_across_ranks and distributed_ready()
        )
        if use_global_candidates:
            candidates, layout = gather_variable_with_grad(features)
            candidate_labels, _ = gather_variable(labels, layout)
        else:
            candidates = features
            candidate_labels = labels
            layout = None

        self.last_metrics = {
            "triplet_local_anchors": features.new_tensor(
                float(features.shape[0])
            ),
            "triplet_global_candidates": features.new_tensor(
                float(candidates.shape[0])
            ),
            "triplet_rank_gather": features.new_tensor(
                float(use_global_candidates)
            ),
        }

        distances = torch.cdist(features.float(), candidates.float(), p=2)
        same_identity = labels[:, None].eq(candidate_labels[None, :])
        if use_global_candidates:
            local_indices = torch.arange(
                features.shape[0], device=features.device
            )
            same_identity[
                local_indices, layout.local_offset + local_indices
            ] = False
        else:
            same_identity.fill_diagonal_(False)
        different_identity = ~labels[:, None].eq(candidate_labels[None, :])

        valid = same_identity.any(dim=1) & different_identity.any(dim=1)
        if not valid.any():
            return features.sum() * 0.0

        hardest_positive = distances.masked_fill(
            ~same_identity, float("-inf")
        ).max(dim=1).values
        hardest_negative = distances.masked_fill(
            ~different_identity, float("inf")
        ).min(dim=1).values

        values = F.margin_ranking_loss(
            hardest_negative[valid],
            hardest_positive[valid],
            torch.ones_like(hardest_negative[valid]),
            margin=self.margin,
            reduction="none",
        )
        if use_global_candidates:
            return globally_normalized_local_sum(values)
        return values.mean()
