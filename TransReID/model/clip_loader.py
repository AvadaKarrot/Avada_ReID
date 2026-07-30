"""Shared loader for the repository's OpenAI CLIP checkpoint."""

from __future__ import annotations

import os

import torch

from model.maple.clip import clip


def load_reid_clip(
    cfg,
    h_resolution: int,
    w_resolution: int,
    vision_stride_size: int,
    *,
    trainer: str = "CoOp",
):
    """Load CLIP from ``MODEL.PRETRAIN_PATH`` or the default cache/download."""

    configured_path = str(getattr(cfg.MODEL, "PRETRAIN_PATH", "")).strip()
    if configured_path:
        if not os.path.isfile(configured_path):
            raise FileNotFoundError(
                f"Configured CLIP checkpoint does not exist: {configured_path}"
            )
        model_path = configured_path
    else:
        model_path = clip._download(clip._MODELS[cfg.MODEL.NAME])

    try:
        scripted_model = torch.jit.load(
            model_path, map_location="cpu"
        ).eval()
        state_dict = scripted_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    maple_cfg = getattr(cfg, "MAPLE", None)
    design_details = {
        "trainer": trainer,
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": int(getattr(maple_cfg, "N_CTX", 0)),
        "person_reid": cfg.INPUT.PERSON_REID,
        "person_img_size": cfg.INPUT.SIZE_TRAIN,
        "vision_stride_size": cfg.MODEL.STRIDE_SIZE,
    }
    model = clip.build_model(
        state_dict,
        design_details,
        h_resolution,
        w_resolution,
        vision_stride_size,
    )
    print(f"Loading pretrained Clip model......from {model_path}")
    return model
