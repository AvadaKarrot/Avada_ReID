import torch
from torch import nn


class ReIDHead(nn.Module):
    """Shared projection, BNNeck, and classifier for all backbones."""

    def __init__(self, input_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.projection = (
            nn.Identity()
            if input_dim == embed_dim
            else nn.Linear(input_dim, embed_dim, bias=False)
        )
        self.bnneck = nn.BatchNorm1d(embed_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.projection, nn.Linear):
            nn.init.kaiming_normal_(self.projection.weight, mode="fan_out")
        nn.init.ones_(self.bnneck.weight)
        nn.init.zeros_(self.bnneck.bias)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, feature: torch.Tensor):
        raw_feature = self.projection(feature)
        embedding = self.bnneck(raw_feature)
        logits = self.classifier(embedding) if self.training else None
        return raw_feature, embedding, logits
