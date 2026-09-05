import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .batch import move_to_device, normalize_batch
from .checkpoint import (
    TrainingCheckpointer,
    TrainingState,
    period_due,
    save_epoch_state,
)


class _DistributedTrainingGraph(nn.Module):
    """Keep model and objective in one DDP forward graph.

    The rank-local batch is deliberately not gathered: this preserves the
    legacy two-GPU semantics for ID, Triplet, Caption, Codebook, and Relation
    losses while DDP averages parameter gradients across ranks.
    """

    def __init__(self, model, objective):
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(self, images, batch):
        return self.objective(self.model(images), batch)


class Trainer:
    """Backbone-agnostic single-device or distributed trainer.

    The model sees images only. Optional captions remain in ``batch`` and are
    consumed exclusively by the objective.
    """

    def __init__(
        self,
        model,
        objective,
        optimizer,
        scheduler=None,
        evaluator=None,
        device="cuda",
        amp_enabled=True,
        amp_init_scale=65536.0,
        output_dir="",
        model_name="reid",
        log_period=100,
        distributed=False,
        local_rank=0,
    ):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.device = torch.device(device)
        self.distributed = bool(distributed)
        self.local_rank = int(local_rank)
        self.rank = dist.get_rank() if self.distributed else 0
        self.is_main_process = self.rank == 0
        self.amp_enabled = amp_enabled and self.device.type == "cuda"
        self.output_dir = Path(output_dir or ".")
        self.log_period = log_period
        self.logger = logging.getLogger("transreid.train")
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.amp_enabled,
            init_scale=amp_init_scale,
        )

        self.model.to(self.device)
        self.objective.to(self.device)
        self.training_graph = _DistributedTrainingGraph(
            self.model, self.objective
        )
        if self.distributed:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Distributed Trainer requires an initialized process group"
                )
            self.training_graph = DistributedDataParallel(
                self.training_graph,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.checkpointer = TrainingCheckpointer(
            self.output_dir,
            model=self.model,
            model_name=model_name,
            objective=self.objective,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )

    def train_step(self, raw_batch):
        batch = normalize_batch(raw_batch)
        images = move_to_device(
            batch["images"], self.device, non_blocking=True
        )
        batch["pids"] = batch["pids"].to(
            self.device, non_blocking=True
        )
        if batch.get("caption_mask") is not None:
            batch["caption_mask"] = batch["caption_mask"].to(
                self.device, non_blocking=True
            )
        if batch.get("attribute_targets") is not None:
            batch["attribute_targets"] = move_to_device(
                batch["attribute_targets"],
                self.device,
                non_blocking=True,
            )

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, enabled=self.amp_enabled
        ):
            losses = self.training_graph(images, batch)

        self.scaler.scale(losses["total"]).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {name: value.detach() for name, value in losses.items()}

    def _mean_metrics_across_ranks(self, losses):
        if not self.distributed:
            return losses
        reduced = {}
        for name, value in losses.items():
            value = value.detach().float().clone()
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            reduced[name] = value / dist.get_world_size()
        return reduced

    def fit(
        self,
        train_loader,
        max_epochs,
        checkpoint_period=10,
        eval_period=0,
        validation=None,
        resume=False,
        resume_path=None,
        max_iterations_per_epoch=0,
        save_checkpoints=True,
    ):
        state = TrainingState()
        if resume:
            state = self.checkpointer.resume(resume_path)
            if self.distributed:
                dist.barrier()
            if self.is_main_process:
                self.logger.info(
                    "Resumed epoch %d; best target mAP %.1f%% at epoch %d",
                    state.epoch,
                    state.best_mAP * 100,
                    state.best_epoch,
                )

        for epoch in range(state.epoch + 1, max_epochs + 1):
            self.training_graph.train()
            if hasattr(self.objective, "set_epoch"):
                self.objective.set_epoch(epoch, max_epochs=max_epochs)

            for iteration, batch in enumerate(train_loader, start=1):
                losses = self.train_step(batch)
                if iteration % self.log_period == 0:
                    losses = self._mean_metrics_across_ranks(losses)
                    summary = ", ".join(
                        f"{name}={value.item():.4f}"
                        for name, value in losses.items()
                    )
                    if self.is_main_process:
                        self.logger.info(
                            "epoch=%d iteration=%d %s",
                            epoch,
                            iteration,
                            summary,
                        )
                if (
                    max_iterations_per_epoch > 0
                    and iteration >= max_iterations_per_epoch
                ):
                    break

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = None
            should_evaluate = (
                validation is not None
                and self.evaluator is not None
                and period_due(epoch, eval_period)
            )
            if should_evaluate and self.is_main_process:
                metrics = self.evaluator.evaluate_loader(
                    self.model,
                    validation["loader"],
                    validation["num_query"],
                )
                self.logger.info(
                    "validation epoch=%d metrics=%s", epoch, metrics
                )

            if self.is_main_process and save_checkpoints:
                is_best = save_epoch_state(
                    checkpointer=self.checkpointer,
                    state=state,
                    epoch=epoch,
                    max_epochs=max_epochs,
                    checkpoint_period=checkpoint_period,
                    metrics=metrics,
                    logger=self.logger,
                )
                if metrics is not None:
                    self.logger.info(
                        "epoch=%d target_mAP=%.4f best_mAP=%.4f "
                        "best_epoch=%d%s",
                        epoch,
                        metrics["mAP"],
                        state.best_mAP,
                        state.best_epoch,
                        " *" if is_best else "",
                    )
            elif self.is_main_process:
                state.epoch = epoch
            else:
                state.epoch = epoch
            if self.distributed:
                dist.barrier()

        return state
