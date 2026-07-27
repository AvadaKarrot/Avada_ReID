import torch
from torch import nn

from .outputs import ReIDOutput


class ReIDModel(nn.Module):
    """Pure image model shared by training and target-domain inference."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images: torch.Tensor) -> ReIDOutput:
        backbone_output = self.backbone.forward_features(images)
        raw_feature, embedding, logits = self.head(
            backbone_output.global_feature
        )
        return ReIDOutput(
            embedding=embedding,
            raw_feature=raw_feature,
            logits=logits,
            patch_features=backbone_output.patch_features,
        )
