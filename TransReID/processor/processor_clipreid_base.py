"""Training and image-only evaluation for the legacy visual CLIP baseline."""

from __future__ import annotations

import logging
import os.path as osp
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from engine.batch import normalize_batch
from utils import CheckpointManager
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


def _is_main_process(distributed: bool) -> bool:
    return not distributed or dist.get_rank() == 0


def _camera_labels(batch, cfg, device):
    if cfg.MODEL.SIE_VIEW:
        raise ValueError(
            "The legacy image batches do not contain view labels; "
            "MODEL.SIE_VIEW must remain False for this baseline"
        )
    if cfg.MODEL.SIE_CAMERA:
        return batch["camids"].to(device, non_blocking=True)
    return None


def _evaluate(model, data_manager, cfg, device, epoch, logger):
    evaluator = R1_mAP_eval(
        data_manager.num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
    )
    evaluator.reset()
    model.eval()

    with torch.no_grad():
        for legacy_batch in data_manager.test_loader:
            batch = normalize_batch(legacy_batch)
            if batch["captions"] is not None:
                raise RuntimeError(
                    "Caption data entered the image-only target evaluation"
                )
            images = batch["images"].to(device, non_blocking=True)
            camera_labels = _camera_labels(batch, cfg, device)
            features = model(images, cam_label=camera_labels)
            evaluator.update(
                (features, batch["pids"], batch["camids"])
            )

    cmc, mean_ap, *_ = evaluator.compute()
    logger.info("Validation Results - Epoch: %s", epoch)
    logger.info("mAP: %.1f%%", mean_ap * 100)
    for rank in (1, 5, 10):
        logger.info("CMC curve, Rank-%-3d:%.1f%%", rank, cmc[rank - 1] * 100)
    return cmc, mean_ap


def _checkpoint_path(cfg, epoch):
    return osp.join(
        cfg.OUTPUT_DIR,
        f"{cfg.MODEL.NAME}_epoch{epoch}.pth",
    )


def _save_checkpoint(manager, cfg, epoch):
    path = _checkpoint_path(cfg, epoch)
    # CheckpointManager expects a zero-based epoch and stores epoch + 1.
    manager.save(epoch=epoch - 1, fpath=path)
    return path


def do_train_clipreid_base(
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
    """Fine-tune the pretrained CLIP visual encoder on the source domain."""

    if cfg.OBJECTIVE.CAPTION.ENABLED or cfg.MODEL.CAPTION:
        raise ValueError(
            "The visual CLIP baseline requires Caption to be disabled"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CLIP-ReID training")

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
    train_logger.info("Start single-stage visual CLIP training")

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=cfg.SOLVER.AMP_ENABLED,
        init_scale=cfg.SOLVER.AMP_INIT_SCALE,
    )
    checkpoint_manager = CheckpointManager(
        logs_dir=cfg.OUTPUT_DIR,
        model=model,
    )
    best_mean_ap = -1.0

    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        epoch_start = time.time()
        loss_meter.reset()
        accuracy_meter.reset()
        model.train()

        for iteration, legacy_batch in enumerate(
            data_manager.train_loader, start=1
        ):
            batch = normalize_batch(legacy_batch)
            if batch["captions"] is not None:
                raise RuntimeError(
                    "Caption data entered the image-only source training"
                )

            images = batch["images"].to(device, non_blocking=True)
            targets = batch["pids"].to(device, non_blocking=True)
            camera_labels = _camera_labels(batch, cfg, device)

            optimizer.zero_grad(set_to_none=True)
            optimizer_center.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", enabled=cfg.SOLVER.AMP_ENABLED
            ):
                scores, features = model(
                    images,
                    targets,
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

            primary_score = scores[0] if isinstance(scores, list) else scores
            accuracy = (
                primary_score.argmax(dim=1) == targets
            ).float().mean()
            loss_meter.update(loss.item(), images.shape[0])
            accuracy_meter.update(accuracy.item(), images.shape[0])

            if iteration % cfg.SOLVER.LOG_PERIOD == 0:
                train_logger.info(
                    "Epoch[%d] Iteration[%d/%d] "
                    "Loss: %.3f, Acc: %.3f, Base Lr: %.2e, AMP: %.0f",
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
        seconds_per_batch = elapsed / max(len(data_manager.train_loader), 1)
        train_logger.info(
            "Epoch %d done. Time per batch: %.3f[s] "
            "Speed: %.1f[samples/s]",
            epoch,
            seconds_per_batch,
            data_manager.train_loader.batch_size / seconds_per_batch,
        )

        scheduler.step()

        if (
            _is_main_process(distributed)
            and epoch % cfg.SOLVER.CHECKPOINT_PERIOD == 0
        ):
            _save_checkpoint(checkpoint_manager, cfg, epoch)

        should_evaluate = (
            cfg.TEST.EVAL
            and epoch % cfg.SOLVER.EVAL_PERIOD == 0
            and _is_main_process(distributed)
        )
        if should_evaluate:
            _, mean_ap = _evaluate(
                model,
                data_manager,
                cfg,
                device,
                epoch,
                test_logger,
            )
            is_best = mean_ap > best_mean_ap
            best_mean_ap = max(best_mean_ap, mean_ap)
            if is_best:
                path = _checkpoint_path(cfg, epoch)
                if not osp.isfile(path):
                    path = _save_checkpoint(
                        checkpoint_manager, cfg, epoch
                    )
                checkpoint_manager.save_best_checkpoint(
                    epoch=epoch - 1,
                    is_best=True,
                    fpath=path,
                )
            train_logger.info(
                "Epoch %d model mAP: %.1f%% best: %.1f%%%s",
                epoch,
                mean_ap * 100,
                best_mean_ap * 100,
                " *" if is_best else "",
            )
            torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    """Evaluate a trained visual baseline using image inputs only."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CLIP-ReID inference")
    device = torch.device("cuda", 0)
    logger = logging.getLogger("transreid.test")
    logger.info("Enter image-only inference")

    evaluator = R1_mAP_eval(
        num_query,
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
    )
    evaluator.reset()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for legacy_batch in val_loader:
            batch = normalize_batch(legacy_batch)
            images = batch["images"].to(device, non_blocking=True)
            camera_labels = _camera_labels(batch, cfg, device)
            features = model(images, cam_label=camera_labels)
            evaluator.update(
                (features, batch["pids"], batch["camids"])
            )

    cmc, mean_ap, *_ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP: %.1f%%", mean_ap * 100)
    for rank in (1, 5, 10):
        logger.info("CMC curve, Rank-%-3d:%.1f%%", rank, cmc[rank - 1] * 100)
    return cmc[0], cmc[4]
