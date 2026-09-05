"""Unified training entry point for CLIP, DINOv3, and SigLIP2 backbones."""

import argparse
import atexit
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import cfg
from data.build import build_datamanager
from engine import Evaluator, Trainer
from modeling import build_model
from objectives import (
    build_attribute_codebook_objective,
    build_attribute_relation_objective,
    build_caption_objective,
    build_objective,
)
from optim import build_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from utils.config_validation import validate_training_config
from utils.logger import setup_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Unified ReID training")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument(
        "--smoke_iterations",
        default=0,
        type=int,
        help=(
            "Run only this many training iterations in one epoch, without "
            "evaluation or checkpoints; intended for memory validation"
        ),
    )
    parser.add_argument(
        "opts",
        help="Override configuration options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    validate_training_config(cfg)
    cfg.freeze()

    distributed = bool(cfg.MODEL.DIST_TRAIN)
    local_rank = int(os.getenv("LOCAL_RANK", args.local_rank))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        atexit.register(
            lambda: dist.destroy_process_group()
            if dist.is_initialized()
            else None
        )
        world_size = dist.get_world_size()
        if cfg.SOLVER.IMS_PER_BATCH % world_size:
            raise ValueError(
                "SOLVER.IMS_PER_BATCH must be divisible by world size"
            )
        local_batch = cfg.SOLVER.IMS_PER_BATCH // world_size
        if local_batch % cfg.DATALOADER.NUM_INSTANCE:
            raise ValueError(
                "Per-rank batch size must be divisible by "
                "DATALOADER.NUM_INSTANCE"
            )
        device = f"cuda:{local_rank}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.MODEL.DEVICE_ID)
        world_size = 1
        local_batch = cfg.SOLVER.IMS_PER_BATCH
        device = cfg.MODEL.DEVICE

    set_seed(cfg.SOLVER.SEED)
    is_main_process = not distributed or dist.get_rank() == 0
    if is_main_process:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    logger = setup_logger(
        "transreid",
        cfg.OUTPUT_DIR if is_main_process else None,
        if_train=True,
    )
    if is_main_process:
        logger.info("Running unified image-only training with config:\n%s", cfg)
        logger.info(
            "Distributed=%s world_size=%d global_batch=%d per_rank_batch=%d",
            distributed,
            world_size,
            cfg.SOLVER.IMS_PER_BATCH,
            local_batch,
        )

    data_manager = build_datamanager(cfg)
    model = build_model(cfg, num_classes=data_manager._num_train_pids)
    caption_objective = build_caption_objective(
        cfg, image_dim=model.head.alignment_dim
    )
    attribute_codebook_objective = build_attribute_codebook_objective(
        cfg, image_dim=model.head.alignment_dim
    )
    attribute_relation_objective = build_attribute_relation_objective(
        cfg, image_dim=model.head.alignment_dim
    )
    objective = build_objective(
        cfg,
        caption_objective=caption_objective,
        attribute_codebook_objective=attribute_codebook_objective,
        attribute_relation_objective=attribute_relation_objective,
    )
    optimizer = build_optimizer(cfg, model, objective)
    scheduler = WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        cfg.SOLVER.WARMUP_FACTOR,
        cfg.SOLVER.WARMUP_ITERS,
        cfg.SOLVER.WARMUP_METHOD,
    )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is configured but unavailable")

    evaluator = Evaluator(
        device=device,
        feat_norm=cfg.TEST.FEAT_NORM == "yes",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        device=device,
        amp_enabled=cfg.SOLVER.AMP_ENABLED,
        amp_init_scale=cfg.SOLVER.AMP_INIT_SCALE,
        output_dir=cfg.OUTPUT_DIR,
        model_name=cfg.MODEL.BACKBONE.NAME,
        log_period=cfg.SOLVER.LOG_PERIOD,
        distributed=distributed,
        local_rank=local_rank,
    )
    smoke_iterations = int(args.smoke_iterations)
    if smoke_iterations < 0:
        raise ValueError("--smoke_iterations must be non-negative")
    validation = None if smoke_iterations else {
        "loader": data_manager.test_loader,
        "num_query": data_manager.num_query,
    }
    try:
        trainer.fit(
            train_loader=data_manager.train_loader,
            max_epochs=1 if smoke_iterations else cfg.SOLVER.MAX_EPOCHS,
            checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD,
            eval_period=cfg.SOLVER.EVAL_PERIOD,
            validation=validation,
            resume=cfg.SOLVER.RESUME_TRAIN,
            resume_path=cfg.SOLVER.RESUME_PATH or None,
            max_iterations_per_epoch=smoke_iterations,
            save_checkpoints=not bool(smoke_iterations),
        )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
