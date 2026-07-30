"""One-batch GPU smoke test for source Caption and image-only target paths."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg
from data.build import build_datamanager
from engine import Trainer, normalize_batch
from modeling import build_model
from objectives import build_caption_objective, build_objective
from optim import build_optimizer
from utils.config_validation import validate_training_config


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test unified Caption training"
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    validate_training_config(cfg)
    if not cfg.OBJECTIVE.CAPTION.ENABLED:
        raise ValueError("This smoke test requires Caption supervision")
    cfg.freeze()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
    if cfg.MODEL.DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but unavailable")

    data_manager = build_datamanager(cfg)
    model = build_model(cfg, num_classes=data_manager._num_train_pids)
    caption_objective = build_caption_objective(
        cfg, image_dim=model.head.embed_dim
    )
    objective = build_objective(
        cfg, caption_objective=caption_objective
    )
    optimizer = build_optimizer(cfg, model, objective)
    trainer = Trainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        device=cfg.MODEL.DEVICE,
        amp_enabled=cfg.SOLVER.AMP_ENABLED,
        amp_init_scale=cfg.SOLVER.AMP_INIT_SCALE,
        output_dir=cfg.OUTPUT_DIR,
        model_name=cfg.MODEL.BACKBONE.NAME,
    )

    source_batch = next(iter(data_manager.train_loader))
    source_named = normalize_batch(source_batch)
    if source_named.get("captions") is None:
        raise RuntimeError("Source smoke batch has no captions")
    losses = trainer.train_step(source_batch)

    target_batch = normalize_batch(
        next(iter(data_manager.test_loader))
    )
    if target_batch.get("captions") is not None:
        raise RuntimeError("Target smoke batch unexpectedly has captions")
    model.eval()
    with torch.no_grad():
        target_outputs = model(
            target_batch["images"].to(trainer.device)
        )

    print(
        json.dumps(
            {
                "source_batch": int(source_named["images"].shape[0]),
                "caption_valid": int(
                    source_named["caption_mask"].sum().item()
                ),
                "losses": {
                    name: float(value.item())
                    for name, value in losses.items()
                },
                "target_batch": int(target_batch["images"].shape[0]),
                "target_embedding_shape": list(
                    target_outputs.embedding.shape
                ),
                "target_caption_input": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
