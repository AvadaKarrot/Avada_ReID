from .reid_objective import ReIDObjective
from .losses import CaptionAlignmentObjective
from .text_encoders import LegacyCLIPTextEncoder, SigLIP2TextEncoder


def _getattr_path(obj, path, default=None):
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return default
        current = getattr(current, name)
    return current


def build_objective(cfg, caption_objective=None) -> ReIDObjective:
    return ReIDObjective(
        triplet_margin=float(_getattr_path(cfg, "SOLVER.MARGIN", 0.3)),
        id_weight=float(_getattr_path(cfg, "MODEL.ID_LOSS_WEIGHT", 1.0)),
        triplet_weight=float(
            _getattr_path(cfg, "MODEL.TRIPLET_LOSS_WEIGHT", 1.0)
        ),
        label_smoothing=float(
            _getattr_path(cfg, "OBJECTIVE.LABEL_SMOOTHING", 0.0)
        ),
        caption_objective=caption_objective,
        caption_weight=float(
            _getattr_path(cfg, "OBJECTIVE.CAPTION.WEIGHT", 0.0)
        ),
    )


def build_caption_objective(cfg, image_dim: int):
    if not _getattr_path(cfg, "OBJECTIVE.CAPTION.ENABLED", False):
        return None

    encoder_name = str(
        _getattr_path(
            cfg, "OBJECTIVE.CAPTION.TEXT_ENCODER", "clip_legacy"
        )
    ).lower()
    if encoder_name not in {"clip_legacy", "siglip2_native"}:
        raise ValueError(
            "Unsupported Caption text encoder "
            f"{encoder_name!r}; expected 'clip_legacy' or "
            "'siglip2_native'"
        )

    feature_level = str(
        _getattr_path(
            cfg, "OBJECTIVE.CAPTION.FEATURE_LEVEL", "global"
        )
    ).lower()
    if feature_level != "global":
        raise ValueError(
            "Only global CLIP CLS-to-EOT Caption alignment is currently "
            "implemented; token-level alignment is a separate ablation"
        )

    trainable = bool(
        _getattr_path(cfg, "OBJECTIVE.CAPTION.TEXT_TRAINABLE", False)
    )
    if encoder_name == "clip_legacy":
        text_encoder = LegacyCLIPTextEncoder(cfg, trainable=trainable)
    else:
        model_name = str(
            _getattr_path(cfg, "MODEL.BACKBONE.PRETRAINED_NAME", "")
        )
        if not model_name:
            raise ValueError(
                "siglip2_native requires MODEL.BACKBONE.PRETRAINED_NAME"
            )
        text_encoder = SigLIP2TextEncoder(
            model_name=model_name,
            trainable=trainable,
        )
    return CaptionAlignmentObjective(
        image_dim=image_dim,
        text_dim=text_encoder.output_dim,
        text_encoder=text_encoder,
        projection_dim=int(
            _getattr_path(
                cfg, "OBJECTIVE.CAPTION.PROJECTION_DIM", 512
            )
        ),
        temperature=float(
            _getattr_path(cfg, "OBJECTIVE.CAPTION.TEMPERATURE", 0.07)
        ),
        positive_mode=str(
            _getattr_path(cfg, "OBJECTIVE.CAPTION.POSITIVE_MODE", "pid")
        ),
        use_projection=bool(
            _getattr_path(
                cfg, "OBJECTIVE.CAPTION.USE_PROJECTION", True
            )
        ),
    )
