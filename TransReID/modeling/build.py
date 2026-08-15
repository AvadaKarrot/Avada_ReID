from .backbones import build_backbone
from .heads import (
    CLIPReIDParityHead,
    MultiBranchParityHead,
    ReIDHead,
    SigLIP2NativePoolerHead,
)
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
            "siglip2_base_patch16_naflex": "siglip2_base_patch16_naflex",
        }
        backbone_name = aliases.get(legacy_name, legacy_name)

    head_type = str(
        _getattr_path(cfg, "MODEL.HEAD.TYPE", "standard")
    ).lower()
    if head_type == "clipreid_parity":
        if backbone_name != "clip_vit_b16":
            raise ValueError(
                "clipreid_parity head requires the clip_vit_b16 backbone"
            )
        # Legacy CLIP-ReID initializes its two classifiers and BNNecks before
        # constructing CLIP. Matching that order also matches seeded sampling.
        head = CLIPReIDParityHead(
            input_dim=768,
            projected_dim=512,
            num_classes=num_classes,
            neck_feature=str(_getattr_path(cfg, "TEST.NECK_FEAT", "before")),
        )
    elif head_type == "multibranch_parity":
        if backbone_name == "clip_vit_b16":
            raise ValueError(
                "clip_vit_b16 must use clipreid_parity; its secondary "
                "projection is already pretrained"
            )
        head = None
    elif head_type == "siglip2_native_pooler":
        if backbone_name not in {
            "siglip2_base_patch16",
            "siglip2_base_patch16_naflex",
        }:
            raise ValueError(
                "siglip2_native_pooler requires a SigLIP2 backbone"
            )
        head = None
    elif head_type != "standard":
        raise ValueError(f"Unsupported ReID head: {head_type}")
    else:
        head = None

    kwargs = {"cfg": cfg} if backbone_name == "clip_vit_b16" else {}
    model_name = _getattr_path(cfg, "MODEL.BACKBONE.PRETRAINED_NAME")
    if model_name and backbone_name != "clip_vit_b16":
        kwargs["model_name"] = model_name
    if backbone_name == "siglip2_base_patch16":
        kwargs["static_position_embedding"] = bool(
            _getattr_path(
                cfg, "MODEL.BACKBONE.STATIC_POSITION_EMBEDDING", False
            )
        )
        kwargs["penultimate_map_pooler"] = bool(
            _getattr_path(
                cfg, "MODEL.BACKBONE.PENULTIMATE_MAP_POOLER", False
            )
        )
        kwargs["target_image_size"] = tuple(cfg.INPUT.SIZE_TRAIN)
    elif backbone_name == "siglip2_base_patch16_naflex":
        if bool(
            _getattr_path(
                cfg, "MODEL.BACKBONE.STATIC_POSITION_EMBEDDING", False
            )
        ):
            raise ValueError(
                "NaFlex uses its native spatial-shape contract; disable "
                "STATIC_POSITION_EMBEDDING"
            )
        kwargs["penultimate_map_pooler"] = bool(
            _getattr_path(
                cfg, "MODEL.BACKBONE.PENULTIMATE_MAP_POOLER", False
            )
        )

    backbone = build_backbone(backbone_name, **kwargs)
    if head is None and head_type == "multibranch_parity":
        secondary_dim = int(
            getattr(backbone, "secondary_dim", backbone.output_dim)
        )
        head = MultiBranchParityHead(
            input_dim=backbone.output_dim,
            secondary_dim=secondary_dim,
            projected_dim=512,
            num_classes=num_classes,
            neck_feature=str(_getattr_path(cfg, "TEST.NECK_FEAT", "before")),
        )
    elif head is None and head_type == "siglip2_native_pooler":
        use_penultimate_metric = bool(
            _getattr_path(
                cfg, "MODEL.BACKBONE.PENULTIMATE_MAP_POOLER", False
            )
        )
        head = SigLIP2NativePoolerHead(
            input_dim=backbone.output_dim,
            pooler_dim=int(
                getattr(backbone, "secondary_dim", backbone.output_dim)
            ),
            num_classes=num_classes,
            neck_feature=str(_getattr_path(cfg, "TEST.NECK_FEAT", "before")),
            use_penultimate_metric=use_penultimate_metric,
        )
    elif head is None:
        embed_dim = int(_getattr_path(cfg, "MODEL.HEAD.EMBED_DIM", 768))
        head = ReIDHead(
            input_dim=backbone.output_dim,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )
    return ReIDModel(backbone=backbone, head=head)
