"""Dependency-free validation for unified training configuration."""


def validate_training_config(current_cfg):
    caption_enabled = bool(current_cfg.OBJECTIVE.CAPTION.ENABLED)
    if current_cfg.MODEL.CAPTION:
        raise ValueError(
            "MODEL.CAPTION belongs to the legacy model-fusion path. "
            "Use OBJECTIVE.CAPTION.ENABLED for source-only supervision."
        )
    if caption_enabled:
        if not current_cfg.OBJECTIVE.CAPTION.FILE:
            raise ValueError(
                "Caption training requires OBJECTIVE.CAPTION.FILE"
            )
        if current_cfg.OBJECTIVE.CAPTION.WEIGHT <= 0:
            raise ValueError(
                "Caption training requires OBJECTIVE.CAPTION.WEIGHT > 0"
            )
        positive_mode = str(
            getattr(
                current_cfg.OBJECTIVE.CAPTION,
                "POSITIVE_MODE",
                "pid",
            )
        ).lower()
        if positive_mode not in {"pid", "instance"}:
            raise ValueError(
                "OBJECTIVE.CAPTION.POSITIVE_MODE must be 'pid' or "
                "'instance'"
            )
    if getattr(current_cfg.MODEL, "SIE_CAMERA", False):
        raise ValueError(
            "Unified cross-domain training does not support source-camera "
            "embeddings; set MODEL.SIE_CAMERA=False"
        )
    if getattr(current_cfg.MODEL, "SIE_VIEW", False):
        raise ValueError(
            "Unified image batches do not provide view labels; "
            "set MODEL.SIE_VIEW=False"
        )
