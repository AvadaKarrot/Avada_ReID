"""Pairwise sigmoid alignment; encoders, masks and gather reuse Caption NCE."""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from objectives.distributed import distributed_ready
from .caption_alignment import CaptionAlignmentObjective


class CaptionSigmoidObjective(CaptionAlignmentObjective):
    """Instance/PID sigmoid, with explicit and non-interchangeable reductions.

    anchor: sum over pairs / number of anchors (SigLIP-style).
    pair_mean: sum over pairs / number of pairs.
    balanced: half positive mean + half negative mean.
    Missing sign groups are omitted, renormalizing the remaining group to one.
    DDP returns local row contributions for the parent's global normalization.
    """

    def __init__(self, *args, sigmoid_reduction="anchor", sigmoid_bias=-10.0,
                 sigmoid_learnable=True, **kwargs):
        super().__init__(*args, **kwargs)
        if sigmoid_reduction not in {"anchor", "pair_mean", "balanced"}:
            raise ValueError("Invalid sigmoid_reduction")
        if not math.isfinite(sigmoid_bias) or not math.isfinite(self.temperature):
            raise ValueError("Sigmoid bias and temperature must be finite")
        self.sigmoid_reduction = sigmoid_reduction
        scale = torch.tensor(math.log(1.0 / self.temperature))
        bias = torch.tensor(float(sigmoid_bias))
        if sigmoid_learnable:
            self.logit_scale = nn.Parameter(scale)
            self.logit_bias = nn.Parameter(bias)
        else:
            self.register_buffer("logit_scale", scale)
            self.register_buffer("logit_bias", bias)

    def forward(self, *args, **kwargs):
        value = super().forward(*args, **kwargs)
        # Keep scalar parameters connected even when all captions are invalid.
        return value + 0.0 * (self.logit_scale + self.logit_bias)

    def _compute_logits(self, images, texts):
        # Stable FP32 logits/BCE under training AMP; cap inverse temperature.
        with torch.autocast(device_type=images.device.type, enabled=False):
            return (
                self.logit_scale.float().clamp(max=math.log(100.0)).exp()
                * (images.float() @ texts.float().t())
                + self.logit_bias.float()
            )

    def _alignment_loss(self, logits, positive_mask, reduction="mean"):
        if reduction not in {"none", "mean"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        positive_mask = positive_mask.bool()
        logits = logits.float()
        positive = F.softplus(-logits).masked_fill(~positive_mask, 0)
        negative = F.softplus(logits).masked_fill(positive_mask, 0)
        if self.sigmoid_reduction == "balanced":
            # Each rank has local anchors but all global candidates.
            counts = torch.stack((
                positive_mask.sum(),
                (~positive_mask).sum(),
                positive_mask.new_tensor(logits.shape[0], dtype=torch.long),
            ))
            if self.gather_across_ranks and distributed_ready():
                dist.all_reduce(counts)
            npos, nneg, nanchors = counts.to(dtype=logits.dtype)
            groups = (npos > 0).to(logits.dtype) + (nneg > 0).to(logits.dtype)
            values = nanchors / groups.clamp_min(1) * (
                positive.sum(1) / npos.clamp_min(1)
                + negative.sum(1) / nneg.clamp_min(1)
            )
        else:
            values = (positive + negative).sum(1)
            if self.sigmoid_reduction == "pair_mean":
                values = values / max(logits.shape[1], 1)
        # Detached metrics must not alter loss scaling or introduce collectives.
        self.last_metrics["caption_sigmoid_scale"] = self.logit_scale.detach().clamp(
            max=math.log(100.0)).exp()
        self.last_metrics["caption_sigmoid_bias"] = self.logit_bias.detach()
        if reduction == "none":
            return values
        return values.mean() if values.numel() else values.sum()
