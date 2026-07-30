from .reid_objective import ReIDObjective
from .losses import CaptionAlignmentObjective
from .text_encoders import LegacyCLIPTextEncoder


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
    if encoder_name != "clip_legacy":
        raise ValueError(
            "Unsupported Caption text encoder "
            f"{encoder_name!r}; expected 'clip_legacy'"
        )

    text_encoder = LegacyCLIPTextEncoder(
        cfg,
        trainable=bool(
            _getattr_path(
                cfg, "OBJECTIVE.CAPTION.TEXT_TRAINABLE", False
            )
        ),
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
    )
