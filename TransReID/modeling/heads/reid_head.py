import torch
from torch import nn

from ..outputs import BackboneOutput, ReIDOutput


class ReIDHead(nn.Module):
    """Shared projection, BNNeck, and classifier for all backbones."""

    def __init__(self, input_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.alignment_dim = embed_dim
        self.metric_dims = (embed_dim,)
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
            alignment_feature=raw_feature,
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
        self.alignment_dim = projected_dim
        self.metric_dims = (input_dim, input_dim, projected_dim)
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
            # CLIP's projected image CLS and projected text EOT features share
            # the native pretrained 512-D image-text embedding space.
            alignment_feature=projected_feature,
        )


class MultiBranchParityHead(nn.Module):
    """CLIP-ReID supervision contract for non-CLIP visual backbones.

    Each adapter supplies a pre-normalization global feature, its primary
    normalized global feature, and a complementary native global feature.
    The complementary feature is projected to 512 dimensions so every
    backbone receives two ID losses, three triplet losses, and produces the
    same 1,280-D evaluation descriptor as the verified CLIP parity baseline.
    """

    def __init__(
        self,
        input_dim: int,
        secondary_dim: int,
        projected_dim: int,
        num_classes: int,
        neck_feature: str = "before",
    ):
        super().__init__()
        if neck_feature not in {"before", "after"}:
            raise ValueError("neck_feature must be 'before' or 'after'")
        self.input_dim = input_dim
        self.secondary_dim = secondary_dim
        self.projected_dim = projected_dim
        self.embed_dim = input_dim
        # The native alignment branch, when supplied by a backbone such as
        # SigLIP2, remains in the backbone hidden dimension. The learned
        # 512-D projection below is ReID-only and must not redefine the
        # pretrained image-text space.
        self.alignment_dim = input_dim
        self.metric_dims = (input_dim, input_dim, projected_dim)
        self.num_classes = num_classes
        self.neck_feature = neck_feature

        self.secondary_projection = nn.Linear(
            secondary_dim, projected_dim, bias=False
        )
        nn.init.kaiming_normal_(
            self.secondary_projection.weight, mode="fan_out"
        )

        self.classifier = nn.Linear(input_dim, num_classes, bias=False)
        CLIPReIDParityHead._init_classifier(self.classifier)
        self.classifier_proj = nn.Linear(
            projected_dim, num_classes, bias=False
        )
        CLIPReIDParityHead._init_classifier(self.classifier_proj)

        self.bnneck = nn.BatchNorm1d(input_dim)
        self.bnneck.bias.requires_grad_(False)
        CLIPReIDParityHead._init_bn(self.bnneck)
        self.bnneck_proj = nn.BatchNorm1d(projected_dim)
        self.bnneck_proj.bias.requires_grad_(False)
        CLIPReIDParityHead._init_bn(self.bnneck_proj)

    def forward(self, backbone_output: BackboneOutput):
        pre_norm_feature = backbone_output.pre_norm_global
        secondary_feature = backbone_output.secondary_global
        if pre_norm_feature is None or secondary_feature is None:
            raise RuntimeError(
                "MultiBranchParityHead requires pre_norm_global and "
                "secondary_global backbone features"
            )

        auxiliary = backbone_output.auxiliary_features or {}
        metric_first_feature = auxiliary.get(
            "penultimate_map_global", pre_norm_feature
        )
        raw_feature = backbone_output.global_feature
        projected_feature = self.secondary_projection(secondary_feature)
        embedding = self.bnneck(raw_feature)
        projected_embedding = self.bnneck_proj(projected_feature)
        alignment_feature = auxiliary.get("alignment_global")

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
                metric_first_feature,
                raw_feature,
                projected_feature,
            ),
            alignment_feature=alignment_feature,
        )


class SigLIP2NativePoolerHead(nn.Module):
    """Use SigLIP2's final pretrained MAP feature for ID and inference.

    An optional penultimate MAP feature can receive an additional triplet
    loss.  It never changes the classifier, caption-alignment feature, or
    evaluation descriptor.
    """

    def __init__(
        self,
        input_dim: int,
        pooler_dim: int,
        num_classes: int,
        neck_feature: str = "before",
        use_penultimate_metric: bool = False,
    ):
        super().__init__()
        if neck_feature not in {"before", "after"}:
            raise ValueError("neck_feature must be 'before' or 'after'")
        self.input_dim = input_dim
        self.pooler_dim = pooler_dim
        self.embed_dim = pooler_dim
        self.alignment_dim = pooler_dim
        self.use_penultimate_metric = bool(use_penultimate_metric)
        self.metric_dims = (
            (pooler_dim, pooler_dim)
            if self.use_penultimate_metric
            else (pooler_dim,)
        )
        self.num_classes = num_classes
        self.neck_feature = neck_feature

        self.classifier_pooler = nn.Linear(pooler_dim, num_classes, bias=False)
        CLIPReIDParityHead._init_classifier(self.classifier_pooler)

        self.bnneck_pooler = nn.BatchNorm1d(pooler_dim)
        self.bnneck_pooler.bias.requires_grad_(False)
        CLIPReIDParityHead._init_bn(self.bnneck_pooler)

    def forward(self, backbone_output: BackboneOutput):
        pooler_feature = backbone_output.secondary_global
        if pooler_feature is None:
            raise RuntimeError(
                "SigLIP2NativePoolerHead requires secondary_global"
            )

        auxiliary = backbone_output.auxiliary_features or {}
        if self.use_penultimate_metric:
            penultimate_feature = auxiliary.get("penultimate_map_global")
            if penultimate_feature is None:
                raise RuntimeError(
                    "SigLIP2NativePoolerHead requires "
                    "penultimate_map_global when its penultimate metric "
                    "branch is enabled"
                )
            metric_features = (penultimate_feature, pooler_feature)
        else:
            metric_features = (pooler_feature,)
        pooler_embedding = self.bnneck_pooler(pooler_feature)
        pooler_logits = None
        if self.training:
            pooler_logits = self.classifier_pooler(pooler_embedding)

        if self.neck_feature == "after":
            evaluation_embedding = pooler_embedding
        else:
            evaluation_embedding = pooler_feature

        return ReIDOutput(
            embedding=evaluation_embedding,
            raw_feature=pooler_feature,
            logits=pooler_logits,
            patch_features=backbone_output.patch_features,
            id_logits=(pooler_logits,) if pooler_logits is not None else None,
            metric_features=metric_features,
            alignment_feature=auxiliary.get("alignment_global", pooler_feature),
        )
