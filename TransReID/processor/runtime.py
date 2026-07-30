"""Shared evaluation, checkpoint, and resume policy for legacy processors."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.distributed as dist

from engine.checkpoint import (
    TrainingCheckpointer,
    TrainingState,
    period_due,
    save_epoch_state,
)
from utils.metrics import R1_mAP_eval


def is_main_process(distributed: bool) -> bool:
    return not distributed or dist.get_rank() == 0


def unwrap_model(model):
    """Avoid DDP collectives during rank-zero-only target evaluation."""

    return model.module if hasattr(model, "module") else model


def synchronize(distributed: bool) -> None:
    if distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def build_checkpointer(
    cfg,
    *,
    model,
    center_criterion,
    optimizer,
    optimizer_center,
    scheduler,
    scaler,
) -> TrainingCheckpointer:
    return TrainingCheckpointer(
        cfg.OUTPUT_DIR,
        model=model,
        model_name=cfg.MODEL.NAME,
        center_criterion=center_criterion,
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        scaler=scaler,
    )


def resume_training(cfg, checkpointer, logger) -> TrainingState:
    if not cfg.SOLVER.RESUME_TRAIN:
        return TrainingState()

    resume_path = cfg.SOLVER.RESUME_PATH or None
    state = checkpointer.resume(resume_path)
    logger.info(
        "Resumed complete training state at epoch %d "
        "(best target mAP %.1f%% at epoch %d)",
        state.epoch,
        state.best_mAP * 100,
        state.best_epoch,
    )
    return state


def evaluate_reid(
    *,
    model,
    loader,
    num_query,
    feat_norm,
    epoch,
    forward_batch: Callable,
    logger: logging.Logger,
):
    """Run one query/gallery evaluation and return serializable metrics."""

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=feat_norm,
    )
    evaluator.reset()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            features, pids, camids = forward_batch(model, batch)
            evaluator.update((features, pids, camids))

    cmc, mean_ap, *_ = evaluator.compute()
    label = f" - Epoch: {epoch}" if epoch is not None else ""
    logger.info("Validation Results%s", label)
    logger.info("mAP: %.1f%%", mean_ap * 100)
    for rank in (1, 5, 10):
        logger.info(
            "CMC curve, Rank-%-3d: %.1f%%",
            rank,
            cmc[rank - 1] * 100,
        )
    return {
        "mAP": float(mean_ap),
        "rank1": float(cmc[0]),
        "rank5": float(cmc[4]),
        "rank10": float(cmc[9]),
    }
