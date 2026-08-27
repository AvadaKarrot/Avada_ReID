"""Dependency-free validation for unified training configuration."""


def validate_training_config(current_cfg):
    caption_enabled = bool(current_cfg.OBJECTIVE.CAPTION.ENABLED)
    attribute_node = getattr(
        current_cfg.OBJECTIVE, "ATTRIBUTE_CODEBOOK", None
    )
    attribute_enabled = bool(
        attribute_node is not None
        and getattr(attribute_node, "ENABLED", False)
    )
    relation_node = getattr(
        current_cfg.OBJECTIVE, "ATTRIBUTE_RELATION", None
    )
    relation_enabled = bool(
        relation_node is not None
        and getattr(relation_node, "ENABLED", False)
    )
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
                backbone_name in {
                    "siglip2_base_patch16",
                    "siglip2_base_patch16_naflex",
                }
                and text_encoder == "siglip2_native"
            ):
                raise ValueError(
                    "multibranch_parity Caption alignment is supported only "
                    "for a SigLIP2 backbone with siglip2_native; its "
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
    if attribute_enabled:
        node = attribute_node
        required = {
            "CAPTION_FILE": node.CAPTION_FILE,
            "PHRASE_BANK": node.PHRASE_BANK,
            "CODEBOOK": node.CODEBOOK,
            "MANIFEST": node.MANIFEST,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Attribute codebook training requires " + ", ".join(missing)
            )
        if node.WEIGHT <= 0:
            raise ValueError(
                "Attribute codebook training requires WEIGHT > 0"
            )
        if node.TEXT_TEMPERATURE <= 0 or node.IMAGE_TEMPERATURE <= 0:
            raise ValueError(
                "Attribute codebook temperatures must be positive"
            )
        if node.MISSING_POLICY not in {"error", "mask"}:
            raise ValueError(
                "Attribute codebook MISSING_POLICY must be 'error' or 'mask'"
            )
        if backbone_name != "siglip2_base_patch16_naflex":
            raise ValueError(
                "PR2 Attribute codebook supervision is restricted to the "
                "SigLIP2 NaFlex alignment space"
            )
    if relation_enabled:
        if not attribute_enabled:
            raise ValueError(
                "Attribute relation requires ATTRIBUTE_CODEBOOK.ENABLED"
            )
        if relation_node.WEIGHT <= 0:
            raise ValueError("Attribute relation requires WEIGHT > 0")
        if relation_node.START_EPOCH < 1:
            raise ValueError("Attribute relation START_EPOCH must be >= 1")
        if relation_node.TEMPERATURE <= 0:
            raise ValueError(
                "Attribute relation TEMPERATURE must be positive"
            )
        if relation_node.DOMAIN_KEY not in {"dataset", "camera", "global"}:
            raise ValueError(
                "Attribute relation DOMAIN_KEY must be dataset, camera, "
                "or global"
            )
        if relation_node.QUEUE_SIZE <= 0 or relation_node.MAX_DOMAINS <= 0:
            raise ValueError(
                "Attribute relation queue dimensions must be positive"
            )
        if (
            relation_node.MIN_CODE_MASS <= 0
            or relation_node.MIN_EFFECTIVE_SAMPLES <= 0
            or relation_node.MIN_ACTIVE_CODES < 2
        ):
            raise ValueError("Invalid Attribute relation coverage thresholds")
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
    if backbone_name == "siglip2_base_patch16_naflex":
        if bool(
            getattr(
                current_cfg.MODEL.BACKBONE,
                "STATIC_POSITION_EMBEDDING",
                False,
            )
        ):
            raise ValueError(
                "SigLIP2 NaFlex must use its native spatial-shape position "
                "contract, not STATIC_POSITION_EMBEDDING"
            )
        if int(current_cfg.MODEL.BACKBONE.NAFLEX_MAX_NUM_PATCHES) <= 0:
            raise ValueError(
                "MODEL.BACKBONE.NAFLEX_MAX_NUM_PATCHES must be positive"
            )
