from torch import nn


class ReIDModel(nn.Module):
    """Pure image model shared by training and target-domain inference."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, images):
        backbone_output = self.backbone.forward_features(images)
        return self.head(backbone_output)
