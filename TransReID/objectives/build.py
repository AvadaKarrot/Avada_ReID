from semantic.artifact import load_torch_artifact

from .losses import (
    AttributeCodebookObjective,
    AttributeRelationObjective,
    CaptionAlignmentObjective,
)
from .reid_objective import ReIDObjective
from .text_encoders import LegacyCLIPTextEncoder, SigLIP2TextEncoder


def _getattr_path(obj, path, default=None):
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return default
        current = getattr(current, name)
    return current


def build_objective(
    cfg,
    caption_objective=None,
    attribute_codebook_objective=None,
    attribute_relation_objective=None,
) -> ReIDObjective:
    return ReIDObjective(
        triplet_margin=float(_getattr_path(cfg, "SOLVER.MARGIN", 0.3)),
        triplet_gather_across_ranks=bool(
            _getattr_path(
                cfg, "OBJECTIVE.TRIPLET.GATHER_ACROSS_RANKS", False
            )
        ),
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
        attribute_codebook_objective=attribute_codebook_objective,
        attribute_codebook_weight=float(
            _getattr_path(cfg, "OBJECTIVE.ATTRIBUTE_CODEBOOK.WEIGHT", 0.0)
        ),
        attribute_codebook_start_epoch=int(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_CODEBOOK.START_EPOCH", 1
            )
        ),
        attribute_codebook_decay_start_epoch=int(
            _getattr_path(
                cfg,
                "OBJECTIVE.ATTRIBUTE_CODEBOOK.DECAY_START_EPOCH",
                0,
            )
        ),
        attribute_codebook_final_weight=float(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_CODEBOOK.FINAL_WEIGHT", 0.0
            )
        ),
        attribute_relation_objective=attribute_relation_objective,
        attribute_relation_weight=float(
            _getattr_path(cfg, "OBJECTIVE.ATTRIBUTE_RELATION.WEIGHT", 0.0)
        ),
        attribute_relation_start_epoch=int(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.START_EPOCH", 6
            )
        ),
    )


def build_attribute_codebook_objective(cfg, image_dim: int):
    if not _getattr_path(
        cfg, "OBJECTIVE.ATTRIBUTE_CODEBOOK.ENABLED", False
    ):
        return None
    payload = load_torch_artifact(
        _getattr_path(cfg, "OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK", "")
    )
    return AttributeCodebookObjective(
        payload["buckets"],
        image_dim=image_dim,
        image_temperature=float(
            _getattr_path(
                cfg,
                "OBJECTIVE.ATTRIBUTE_CODEBOOK.IMAGE_TEMPERATURE",
                0.07,
            )
        ),
        confidence_weighting=bool(
            _getattr_path(
                cfg,
                "OBJECTIVE.ATTRIBUTE_CODEBOOK.CONFIDENCE_WEIGHTING",
                True,
            )
        ),
    )


def build_attribute_relation_objective(cfg, image_dim: int):
    if not _getattr_path(
        cfg, "OBJECTIVE.ATTRIBUTE_RELATION.ENABLED", False
    ):
        return None
    payload = load_torch_artifact(
        _getattr_path(cfg, "OBJECTIVE.ATTRIBUTE_CODEBOOK.CODEBOOK", "")
    )
    return AttributeRelationObjective(
        payload["buckets"],
        image_dim=image_dim,
        relation_temperature=float(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.TEMPERATURE", 0.1
            )
        ),
        domain_key=str(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.DOMAIN_KEY", "dataset"
            )
        ).lower(),
        queue_size=int(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.QUEUE_SIZE", 256
            )
        ),
        max_domains=int(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.MAX_DOMAINS", 32
            )
        ),
        min_code_mass=float(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.MIN_CODE_MASS", 1.0
            )
        ),
        min_effective_samples=float(
            _getattr_path(
                cfg,
                "OBJECTIVE.ATTRIBUTE_RELATION.MIN_EFFECTIVE_SAMPLES",
                2.0,
            )
        ),
        min_active_codes=int(
            _getattr_path(
                cfg, "OBJECTIVE.ATTRIBUTE_RELATION.MIN_ACTIVE_CODES", 3
            )
        ),
        min_anchor_confidence=float(
            _getattr_path(
                cfg,
                "OBJECTIVE.ATTRIBUTE_RELATION.MIN_ANCHOR_CONFIDENCE",
                0.0,
            )
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
    loss_type = str(_getattr_path(cfg, "OBJECTIVE.CAPTION.LOSS_TYPE", "nce")).lower()
    options = {}
    if loss_type == "nce":
        objective_class = CaptionAlignmentObjective
    elif loss_type == "sigmoid":
        from .losses.caption_sigmoid import CaptionSigmoidObjective
        objective_class = CaptionSigmoidObjective
        options = dict(
            sigmoid_reduction=_getattr_path(
                cfg, "OBJECTIVE.CAPTION.SIGMOID_REDUCTION", "anchor"),
            sigmoid_bias=float(_getattr_path(
                cfg, "OBJECTIVE.CAPTION.SIGMOID_BIAS", -10.0)),
            sigmoid_learnable=bool(_getattr_path(
                cfg, "OBJECTIVE.CAPTION.SIGMOID_LEARNABLE", True)),
        )
    else:
        raise ValueError("OBJECTIVE.CAPTION.LOSS_TYPE must be nce or sigmoid")
    return objective_class(
        **options,
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
        gather_across_ranks=bool(
            _getattr_path(
                cfg, "OBJECTIVE.CAPTION.GATHER_ACROSS_RANKS", False
            )
        ),
    )
