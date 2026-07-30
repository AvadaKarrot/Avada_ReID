import logging
from pathlib import Path

import torch

from .batch import normalize_batch
from .checkpoint import (
    TrainingCheckpointer,
    TrainingState,
    period_due,
    save_epoch_state,
)


class Trainer:
    """Backbone-agnostic single-device trainer.

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
    ):
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.device = torch.device(device)
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
        images = batch["images"].to(self.device, non_blocking=True)
        batch["pids"] = batch["pids"].to(
            self.device, non_blocking=True
        )
        if batch.get("caption_mask") is not None:
            batch["caption_mask"] = batch["caption_mask"].to(
                self.device, non_blocking=True
            )

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type, enabled=self.amp_enabled
        ):
            outputs = self.model(images)
            losses = self.objective(outputs, batch)

        self.scaler.scale(losses["total"]).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {name: value.detach() for name, value in losses.items()}

    def fit(
        self,
        train_loader,
        max_epochs,
        checkpoint_period=10,
        eval_period=0,
        validation=None,
        resume=False,
        resume_path=None,
    ):
        state = TrainingState()
        if resume:
            state = self.checkpointer.resume(resume_path)
            self.logger.info(
                "Resumed epoch %d; best target mAP %.1f%% at epoch %d",
                state.epoch,
                state.best_mAP * 100,
                state.best_epoch,
            )

        for epoch in range(state.epoch + 1, max_epochs + 1):
            self.model.train()
            self.objective.train()

            for iteration, batch in enumerate(train_loader, start=1):
                losses = self.train_step(batch)
                if iteration % self.log_period == 0:
                    summary = ", ".join(
                        f"{name}={value.item():.4f}"
                        for name, value in losses.items()
                    )
                    self.logger.info(
                        "epoch=%d iteration=%d %s",
                        epoch,
                        iteration,
                        summary,
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = None
            if (
                validation is not None
                and self.evaluator is not None
                and period_due(epoch, eval_period)
            ):
                metrics = self.evaluator.evaluate_loader(
                    self.model,
                    validation["loader"],
                    validation["num_query"],
                )
                self.logger.info(
                    "validation epoch=%d metrics=%s", epoch, metrics
                )

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

        return state
