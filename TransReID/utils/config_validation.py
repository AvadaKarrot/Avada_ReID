"""Dependency-free validation for unified training configuration."""


def validate_training_config(current_cfg):
    caption_enabled = bool(current_cfg.OBJECTIVE.CAPTION.ENABLED)
    model_head = getattr(current_cfg.MODEL, "HEAD", None)
    head_type = str(getattr(model_head, "TYPE", "standard")).lower()
    backbone_node = getattr(current_cfg.MODEL, "BACKBONE", None)
    backbone_name = str(
        getattr(backbone_node, "NAME", "clip_vit_b16")
    ).lower()
    if current_cfg.MODEL.CAPTION:
        raise ValueError(
            "MODEL.CAPTION belongs to the legacy model-fusion path. "
            "Use OBJECTIVE.CAPTION.ENABLED for source-only supervision."
        )
    if caption_enabled:
        if head_type == "multibranch_parity":
            text_encoder = str(
                getattr(
                    current_cfg.OBJECTIVE.CAPTION,
                    "TEXT_ENCODER",
                    "clip_legacy",
                )
            ).lower()
            if not (
                backbone_name == "siglip2_base_patch16"
                and text_encoder == "siglip2_native"
            ):
                raise ValueError(
                    "multibranch_parity Caption alignment is supported only "
                    "for siglip2_base_patch16 with siglip2_native; its "
                    "512-D ReID projection is never an alignment space"
                )
        if not current_cfg.OBJECTIVE.CAPTION.FILE:
            raise ValueError(
                "Caption training requires OBJECTIVE.CAPTION.FILE"
            )
        if current_cfg.OBJECTIVE.CAPTION.WEIGHT <= 0:
            raise ValueError(
                "Caption training requires OBJECTIVE.CAPTION.WEIGHT > 0"
            )
        feature_level = str(
            getattr(
                current_cfg.OBJECTIVE.CAPTION,
                "FEATURE_LEVEL",
                "global",
            )
        ).lower()
        if feature_level != "global":
            raise ValueError(
                "OBJECTIVE.CAPTION.FEATURE_LEVEL must be 'global'; "
                "token-level alignment is a separate experiment"
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
