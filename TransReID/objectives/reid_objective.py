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
        triplet_gather_across_ranks: bool = False,
        id_weight: float = 1.0,
        triplet_weight: float = 1.0,
        label_smoothing: float = 0.0,
        caption_objective: Optional[nn.Module] = None,
        caption_weight: float = 0.0,
        attribute_codebook_objective: Optional[nn.Module] = None,
        attribute_codebook_weight: float = 0.0,
        attribute_codebook_start_epoch: int = 1,
        attribute_codebook_decay_start_epoch: int = 0,
        attribute_codebook_final_weight: float = 0.0,
        attribute_relation_objective: Optional[nn.Module] = None,
        attribute_relation_weight: float = 0.0,
        attribute_relation_start_epoch: int = 6,
    ):
        super().__init__()
        self.triplet = BatchHardTripletLoss(
            margin=triplet_margin,
            gather_across_ranks=triplet_gather_across_ranks,
        )
        self.id_weight = id_weight
        self.triplet_weight = triplet_weight
        self.label_smoothing = label_smoothing
        self.caption_objective = caption_objective
        self.caption_weight = caption_weight
        self.attribute_codebook_objective = attribute_codebook_objective
        self.attribute_codebook_weight = attribute_codebook_weight
        self.attribute_codebook_start_epoch = int(
            attribute_codebook_start_epoch
        )
        self.attribute_codebook_decay_start_epoch = int(
            attribute_codebook_decay_start_epoch
        )
        self.attribute_codebook_final_weight = float(
            attribute_codebook_final_weight
        )
        self.attribute_relation_objective = attribute_relation_objective
        self.attribute_relation_weight = float(attribute_relation_weight)
        self.attribute_relation_start_epoch = int(
            attribute_relation_start_epoch
        )
        self.current_epoch = 1
        self.max_epochs = 1
        if self.attribute_codebook_start_epoch < 1:
            raise ValueError("attribute_codebook_start_epoch must be >= 1")
        if self.attribute_codebook_decay_start_epoch < 0:
            raise ValueError(
                "attribute_codebook_decay_start_epoch must be >= 0"
            )
        if self.attribute_codebook_final_weight < 0:
            raise ValueError(
                "attribute_codebook_final_weight must be non-negative"
            )
        if self.attribute_relation_weight < 0:
            raise ValueError("attribute_relation_weight must be non-negative")
        if self.attribute_relation_start_epoch < 1:
            raise ValueError("attribute_relation_start_epoch must be >= 1")

    def set_epoch(self, epoch: int, max_epochs: Optional[int] = None):
        self.current_epoch = int(epoch)
        if max_epochs is not None:
            self.max_epochs = int(max_epochs)

    def current_attribute_codebook_weight(self) -> float:
        if self.current_epoch < self.attribute_codebook_start_epoch:
            return 0.0
        decay_start = self.attribute_codebook_decay_start_epoch
        if decay_start <= 0 or self.current_epoch <= decay_start:
            return self.attribute_codebook_weight
        if self.max_epochs <= decay_start:
            return self.attribute_codebook_final_weight
        progress = (self.current_epoch - decay_start) / (
            self.max_epochs - decay_start
        )
        progress = min(max(progress, 0.0), 1.0)
        return (
            self.attribute_codebook_weight
            + progress
            * (
                self.attribute_codebook_final_weight
                - self.attribute_codebook_weight
            )
        )

    def load_state_dict(self, state_dict, strict=True):
        """Load compact objectives while preserving strict validation.

        Frozen caption text weights may be absent because they are rebuilt
        from the configured pretrained directory before checkpoint restore.
        Every other missing or unexpected key remains a hard error.
        """

        incompatible = super().load_state_dict(state_dict, strict=False)
        ignored_prefixes = ()
        if (
            self.caption_objective is not None
            and hasattr(
                self.caption_objective,
                "frozen_text_encoder_state_is_reconstructible",
            )
            and self.caption_objective.
            frozen_text_encoder_state_is_reconstructible()
        ):
            ignored_prefixes = ("caption_objective.text_encoder.",)

        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(ignored_prefixes)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"Missing key(s): {missing}")
            if unexpected:
                details.append(f"Unexpected key(s): {unexpected}")
            raise RuntimeError(
                "Error(s) in loading state_dict for ReIDObjective: "
                + "; ".join(details)
            )
        return type(incompatible)(missing, unexpected)

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
        for name, value in getattr(self.triplet, "last_metrics", {}).items():
            losses[name] = value

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
            for name, value in getattr(
                self.caption_objective, "last_metrics", {}
            ).items():
                losses[name] = value
            total = total + self.caption_weight * losses["caption"]

        codebook_weight = self.current_attribute_codebook_weight()
        if (
            self.attribute_codebook_objective is not None
            and codebook_weight > 0
        ):
            alignment_feature = (
                outputs.alignment_feature
                if outputs.alignment_feature is not None
                else outputs.raw_feature
            )
            losses["attribute_codebook"] = self.attribute_codebook_objective(
                alignment_feature,
                batch.get("attribute_targets"),
            )
            total = total + codebook_weight * losses["attribute_codebook"]
            losses["attribute_codebook_weight"] = total.new_tensor(
                codebook_weight
            )
            for name, value in getattr(
                self.attribute_codebook_objective,
                "last_metrics",
                {},
            ).items():
                losses[name] = value

        if (
            self.attribute_relation_objective is not None
            and self.attribute_relation_weight > 0
            and self.current_epoch >= self.attribute_relation_start_epoch
        ):
            alignment_feature = (
                outputs.alignment_feature
                if outputs.alignment_feature is not None
                else outputs.raw_feature
            )
            losses["attribute_relation"] = (
                self.attribute_relation_objective(
                    alignment_feature,
                    batch.get("attribute_targets"),
                    batch,
                )
            )
            total = total + (
                self.attribute_relation_weight
                * losses["attribute_relation"]
            )
            losses["attribute_relation_weight"] = total.new_tensor(
                self.attribute_relation_weight
            )
            for name, value in getattr(
                self.attribute_relation_objective,
                "last_metrics",
                {},
            ).items():
                losses[name] = value

        losses["total"] = total
        return losses
