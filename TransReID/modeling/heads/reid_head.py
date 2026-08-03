import torch
from torch import nn

from ..outputs import BackboneOutput, ReIDOutput


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

    def forward(self, backbone_output: BackboneOutput):
        feature = backbone_output.global_feature
        raw_feature = self.projection(feature)
        embedding = self.bnneck(raw_feature)
        logits = self.classifier(embedding) if self.training else None
        return ReIDOutput(
            embedding=embedding,
            raw_feature=raw_feature,
            logits=logits,
            patch_features=backbone_output.patch_features,
            id_logits=(logits,) if logits is not None else None,
            metric_features=(raw_feature,),
        )


class CLIPReIDParityHead(nn.Module):
    """Legacy CLIP-ReID dual head behind the unified model contract.

    The legacy baseline applies ID loss to the 768-D visual feature and the
    512-D projected feature, triplet loss to three visual outputs, and uses
    their 1,280-D concatenation for target-domain evaluation.
    """

    def __init__(
        self,
        input_dim: int,
        projected_dim: int,
        num_classes: int,
        neck_feature: str = "before",
    ):
        super().__init__()
        if neck_feature not in {"before", "after"}:
            raise ValueError("neck_feature must be 'before' or 'after'")
        self.input_dim = input_dim
        self.projected_dim = projected_dim
        self.embed_dim = input_dim
        self.num_classes = num_classes
        self.neck_feature = neck_feature

        # Preserve the legacy module construction and RNG-consumption order.
        self.classifier = nn.Linear(input_dim, num_classes, bias=False)
        self._init_classifier(self.classifier)
        self.classifier_proj = nn.Linear(
            projected_dim, num_classes, bias=False
        )
        self._init_classifier(self.classifier_proj)
        self.bnneck = nn.BatchNorm1d(input_dim)
        self.bnneck.bias.requires_grad_(False)
        self._init_bn(self.bnneck)
        self.bnneck_proj = nn.BatchNorm1d(projected_dim)
        self.bnneck_proj.bias.requires_grad_(False)
        self._init_bn(self.bnneck_proj)

    @staticmethod
    def _init_classifier(module):
        nn.init.normal_(module.weight, std=0.001)

    @staticmethod
    def _init_bn(module):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    def forward(self, backbone_output: BackboneOutput):
        auxiliary = backbone_output.auxiliary_features or {}
        try:
            last_feature = auxiliary["last_global"]
            projected_feature = auxiliary["projected_global"]
        except KeyError as error:
            raise RuntimeError(
                "CLIP parity head requires last_global and "
                "projected_global backbone features"
            ) from error

        raw_feature = backbone_output.global_feature
        embedding = self.bnneck(raw_feature)
        projected_embedding = self.bnneck_proj(projected_feature)
        logits = None
        projected_logits = None
        if self.training:
            logits = self.classifier(embedding)
            projected_logits = self.classifier_proj(projected_embedding)

        if self.neck_feature == "after":
            evaluation_embedding = torch.cat(
                (embedding, projected_embedding), dim=1
            )
        else:
            evaluation_embedding = torch.cat(
                (raw_feature, projected_feature), dim=1
            )

        return ReIDOutput(
            embedding=evaluation_embedding,
            raw_feature=raw_feature,
            logits=logits,
            patch_features=backbone_output.patch_features,
            id_logits=(logits, projected_logits)
            if logits is not None
            else None,
            metric_features=(
                last_feature,
                raw_feature,
                projected_feature,
            ),
        )
