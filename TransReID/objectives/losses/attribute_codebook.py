"""Attribute semantic-code distribution distillation for image features."""

import torch
from torch import nn
from torch.nn import functional as F

from data.attribute_store import ATTRIBUTE_BUCKETS


class AttributeCodebookObjective(nn.Module):
    """Distil frozen text-code assignments into NaFlex image features."""

    def __init__(
        self,
        buckets,
        *,
        image_dim,
        image_temperature=0.07,
        confidence_weighting=True,
    ):
        super().__init__()
        if image_temperature <= 0:
            raise ValueError("image_temperature must be positive")
        self.image_temperature = float(image_temperature)
        self.confidence_weighting = bool(confidence_weighting)
        self.bucket_names = tuple(
            bucket for bucket in ATTRIBUTE_BUCKETS if bucket in buckets
        )
        for index, bucket in enumerate(self.bucket_names):
            prototypes = F.normalize(
                buckets[bucket]["prototypes"].float(), dim=1
            )
            if prototypes.shape[1] != int(image_dim):
                raise ValueError(
                    f"{bucket} code dimension {prototypes.shape[1]} does not "
                    f"match image alignment dimension {image_dim}"
                )
            confidence = buckets[bucket].get("anchor_confidence")
            if confidence is None:
                confidence = torch.ones(prototypes.shape[0])
            confidence = confidence.float().clamp_min(0)
            self.register_buffer(
                f"prototype_{index}", prototypes, persistent=False
            )
            self.register_buffer(
                f"confidence_{index}", confidence, persistent=False
            )

    def forward(self, image_features, attribute_targets):
        if attribute_targets is None:
            return image_features.sum() * 0.0
        images = F.normalize(image_features.float(), dim=1)
        quality = attribute_targets["quality"].to(images.device).float()
        weighted_loss = images.new_zeros(())
        weight_sum = images.new_zeros(())
        for index, bucket in enumerate(self.bucket_names):
            target = attribute_targets["distributions"][bucket].to(
                images.device
            ).float()
            mask = attribute_targets["mask"][bucket].to(images.device).bool()
            if not mask.any():
                continue
            prototypes = getattr(self, f"prototype_{index}")
            log_q_image = F.log_softmax(
                images @ prototypes.t() / self.image_temperature, dim=1
            )
            per_sample = F.kl_div(
                log_q_image,
                target,
                reduction="none",
            ).sum(dim=1)
            weights = quality
            if self.confidence_weighting:
                confidence = getattr(self, f"confidence_{index}")
                weights = weights * (target @ confidence)
            weights = weights * mask.float()
            weighted_loss = weighted_loss + (per_sample * weights).sum()
            weight_sum = weight_sum + weights.sum()
        if weight_sum.item() == 0:
            return image_features.sum() * 0.0
        return weighted_loss / weight_sum.clamp_min(1e-12)
