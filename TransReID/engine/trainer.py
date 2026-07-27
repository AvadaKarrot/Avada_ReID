import logging
from pathlib import Path

import torch

from .batch import normalize_batch
from .checkpoint import save_checkpoint


class Trainer:
    """Backbone-agnostic training loop.

    The model sees only images. Optional captions are consumed by the
    objective, preserving an image-only inference contract.
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
        output_dir="",
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
            self.device.type, enabled=self.amp_enabled
        )

        self.model.to(self.device)
        self.objective.to(self.device)

    def train_step(self, raw_batch):
        batch = normalize_batch(raw_batch)
        images = batch["images"].to(self.device)
        batch["pids"] = batch["pids"].to(self.device)
        if batch.get("caption_mask") is not None:
            batch["caption_mask"] = batch["caption_mask"].to(self.device)

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
    ):
        best_map = float("-inf")
        for epoch in range(1, max_epochs + 1):
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
                        "epoch=%d iteration=%d %s", epoch, iteration, summary
                    )

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = None
            if (
                validation is not None
                and self.evaluator is not None
                and eval_period > 0
                and epoch % eval_period == 0
            ):
                metrics = self.evaluator.evaluate_loader(
                    self.model,
                    validation["loader"],
                    validation["num_query"],
                )
                best_map = max(best_map, metrics["mAP"])
                self.logger.info("validation epoch=%d metrics=%s", epoch, metrics)

            if checkpoint_period > 0 and epoch % checkpoint_period == 0:
                save_checkpoint(
                    self.output_dir / f"model_epoch_{epoch}.pth",
                    model=self.model,
                    objective=self.objective,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    extra={"metrics": metrics, "best_mAP": best_map},
                )
