from .backbones import build_backbone
from .heads import ReIDHead
from .reid_model import ReIDModel


def _getattr_path(obj, path, default=None):
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return default
        current = getattr(current, name)
    return current


def build_model(cfg, num_classes: int) -> ReIDModel:
    """Build a backbone-agnostic image ReID model from configuration."""

    backbone_name = _getattr_path(cfg, "MODEL.BACKBONE.NAME")
    if backbone_name is None:
        legacy_name = _getattr_path(cfg, "MODEL.NAME", "ViT-B-16")
        aliases = {
            "ViT-B-16": "clip_vit_b16",
            "dinov3_vit_b16": "dinov3_vit_b16",
            "siglip2_base_patch16": "siglip2_base_patch16",
        }
        backbone_name = aliases.get(legacy_name, legacy_name)

    kwargs = {"cfg": cfg} if backbone_name == "clip_vit_b16" else {}
    model_name = _getattr_path(cfg, "MODEL.BACKBONE.PRETRAINED_NAME")
    if model_name and backbone_name != "clip_vit_b16":
        kwargs["model_name"] = model_name

    backbone = build_backbone(backbone_name, **kwargs)
    embed_dim = int(_getattr_path(cfg, "MODEL.HEAD.EMBED_DIM", 768))
    head = ReIDHead(
        input_dim=backbone.output_dim,
        embed_dim=embed_dim,
        num_classes=num_classes,
    )
    return ReIDModel(backbone=backbone, head=head)
