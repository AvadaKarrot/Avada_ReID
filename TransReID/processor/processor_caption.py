"""Caption-assisted source training with image-only target evaluation."""

from __future__ import annotations

import logging
import time

import torch
from torch.nn.parallel import DistributedDataParallel

from engine.batch import normalize_batch
from processor.runtime import (
    build_checkpointer,
    evaluate_reid,
    is_main_process,
    period_due,
    resume_training,
    save_epoch_state,
    synchronize,
    unwrap_model,
)
from utils.meter import AverageMeter


def _camera_labels(batch, cfg, device):
    if cfg.MODEL.SIE_CAMERA:
        return batch["camids"].to(device, non_blocking=True)
    return None


def _caption_batch(legacy_batch):
    batch = normalize_batch(legacy_batch)
    captions = batch.get("captions")
    if captions is None:
        raise RuntimeError(
            "Caption training requires a caption-enabled source DataLoader"
        )
    caption_mask = batch.get("caption_mask")
    if caption_mask is not None and not bool(caption_mask.all()):
        raise RuntimeError(
            "The source batch contains missing captions; use complete "
            "coverage or an objective that explicitly supports masking"
        )
    return batch


def _forward_image_only_eval(model, legacy_batch, cfg, device):
    batch = normalize_batch(legacy_batch)
    if batch.get("captions") is not None:
        raise RuntimeError(
            "Target evaluation must not receive caption side information"
        )
    images = batch["images"].to(device, non_blocking=True)
    features = model(
        images,
        captions=None,
        cam_label=_camera_labels(batch, cfg, device),
    )
    return features, batch["pids"], batch["camids"]


def do_train_caption(
    cfg,
    model,
    center_criterion,
    data_manager,
    optimizer,
    optimizer_center,
    scheduler,
    loss_fn,
    local_rank,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for caption training")

    distributed = bool(cfg.MODEL.DIST_TRAIN)
    device = torch.device("cuda", local_rank)
    model = model.to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=True,
        )

    train_logger = logging.getLogger("transreid.train")
    test_logger = logging.getLogger("transreid.test")
    train_logger.info(
        "Start caption-assisted source training with image-only target eval"
    )

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.SOLVER.AMP_ENABLED,
        init_scale=cfg.SOLVER.AMP_INIT_SCALE,
    )
    checkpointer = build_checkpointer(
        cfg,
        model=model,
        center_criterion=center_criterion,
        optimizer=optimizer,
        optimizer_center=optimizer_center,
        scheduler=scheduler,
        scaler=scaler,
    )
    state = resume_training(cfg, checkpointer, train_logger)

    for epoch in range(state.epoch + 1, cfg.SOLVER.MAX_EPOCHS + 1):
        epoch_start = time.time()
        loss_meter.reset()
        accuracy_meter.reset()
        model.train()

        for iteration, legacy_batch in enumerate(
            data_manager.train_loader, start=1
        ):
            batch = _caption_batch(legacy_batch)
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["pids"].to(device, non_blocking=True)
            captions = batch["captions"]
            camera_labels = _camera_labels(batch, cfg, device)

            optimizer.zero_grad(set_to_none=True)
            optimizer_center.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=cfg.SOLVER.AMP_ENABLED
            ):
                scores, features = model(
                    images,
                    targets,
                    captions=captions,
                    cam_label=camera_labels,
                )
                loss = loss_fn(
                    scores,
                    features,
                    targets,
                    camera_labels,
                )

            scaler.scale(loss).backward()
            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                scaler.unscale_(optimizer_center)
                for parameter in center_criterion.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(
                            1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                        )
                scaler.step(optimizer_center)
            scaler.step(optimizer)
            scaler.update()

            primary_score = (
                scores[0] if isinstance(scores, list) else scores
            )
            accuracy = (
                primary_score.argmax(dim=1) == targets
            ).float().mean()
            loss_meter.update(loss.item(), images.shape[0])
            accuracy_meter.update(accuracy.item(), images.shape[0])

            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                train_logger.info(
                    "Epoch[%d] Iteration[%d/%d] Loss: %.3f, "
                    "Acc: %.3f, Base Lr: %.2e, AMP: %.0f",
                    epoch,
                    iteration,
                    len(data_manager.train_loader),
                    loss_meter.avg,
                    accuracy_meter.avg,
                    scheduler.get_last_lr()[0],
                    scaler.get_scale(),
                )

        torch.cuda.synchronize(device)
        elapsed = time.time() - epoch_start
        seconds_per_batch = elapsed / max(
            len(data_manager.train_loader), 1
        )
        if is_main_process(distributed):
            train_logger.info(
                "Epoch %d done. Time per batch: %.3f[s] "
                "Speed: %.1f[samples/s]",
                epoch,
                seconds_per_batch,
                data_manager.train_loader.batch_size / seconds_per_batch,
            )
        scheduler.step()

        metrics = None
        if (
            period_due(epoch, cfg.SOLVER.EVAL_PERIOD)
            and is_main_process(distributed)
        ):
            metrics = evaluate_reid(
                model=unwrap_model(model),
                loader=data_manager.test_loader,
                num_query=data_manager.num_query,
                feat_norm=cfg.TEST.FEAT_NORM,
                epoch=epoch,
                forward_batch=lambda current_model, batch: (
                    _forward_image_only_eval(
                        current_model, batch, cfg, device
                    )
                ),
                logger=test_logger,
            )
            torch.cuda.empty_cache()

        if is_main_process(distributed):
            is_best = save_epoch_state(
                checkpointer=checkpointer,
                state=state,
                epoch=epoch,
                max_epochs=cfg.SOLVER.MAX_EPOCHS,
                checkpoint_period=cfg.SOLVER.CHECKPOINT_PERIOD,
                metrics=metrics,
                logger=train_logger,
            )
            if metrics is not None:
                train_logger.info(
                    "Epoch %d target mAP: %.1f%%; best: %.1f%% "
                    "(epoch %d)%s",
                    epoch,
                    metrics["mAP"] * 100,
                    state.best_mAP * 100,
                    state.best_epoch,
                    " *" if is_best else "",
                )
        synchronize(distributed)


def do_inference(cfg, model, val_loader, num_query):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference")
    device = torch.device("cuda", 0)
    logger = logging.getLogger("transreid.test")
    model = model.to(device)
    metrics = evaluate_reid(
        model=model,
        loader=val_loader,
        num_query=num_query,
        feat_norm=cfg.TEST.FEAT_NORM,
        epoch=None,
        forward_batch=lambda current_model, batch: (
            _forward_image_only_eval(current_model, batch, cfg, device)
        ),
        logger=logger,
    )
    return metrics["rank1"], metrics["rank5"]
