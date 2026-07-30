"""Training checkpoint persistence shared by all training entry points."""

from __future__ import annotations

import os
import shutil
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn.parallel import DistributedDataParallel


@dataclass
class TrainingState:
    """State that affects checkpoint selection and resume behavior."""

    epoch: int = 0
    best_mAP: float = float("-inf")
    best_epoch: int = 0

    def update_best(self, mean_ap: float, epoch: int) -> bool:
        """Keep the earlier epoch when two checkpoints have equal mAP."""

        if mean_ap <= self.best_mAP:
            return False
        self.best_mAP = float(mean_ap)
        self.best_epoch = int(epoch)
        return True


def _unwrap_model(model):
    if isinstance(
        model, (torch.nn.DataParallel, DistributedDataParallel)
    ):
        return model.module
    return model


def _state_dict(component):
    if component is None:
        return None
    return _unwrap_model(component).state_dict()


def _load_state_dict(component, state_dict, *, strict=True):
    if component is None or state_dict is None:
        return
    target = _unwrap_model(component)
    if isinstance(target, torch.nn.Module):
        target.load_state_dict(state_dict, strict=strict)
    else:
        target.load_state_dict(state_dict)


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    """Prefer a same-volume hard link for immutable checkpoint aliases."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        os.link(source, temporary_path)
    except OSError:
        shutil.copy2(source, temporary_path)
    os.replace(temporary_path, destination)


class TrainingCheckpointer:
    """Save complete, resumable training checkpoints.

    Evaluated epochs are immutable files. ``checkpoint_latest.pth.tar`` is
    refreshed after every scheduled save, ``model_best.pth.tar`` tracks the
    highest target mAP, and ``model_last.pth.tar`` is written at the end.
    """

    FORMAT_VERSION = 2

    def __init__(
        self,
        output_dir,
        *,
        model,
        model_name,
        objective=None,
        center_criterion=None,
        optimizer=None,
        optimizer_center=None,
        scheduler=None,
        scaler=None,
    ):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.components = {
            "model": model,
            "objective": objective,
            "center_criterion": center_criterion,
            "optimizer": optimizer,
            "optimizer_center": optimizer_center,
            "scheduler": scheduler,
            "scaler": scaler,
        }

    @property
    def latest_path(self) -> Path:
        return self.output_dir / "checkpoint_latest.pth.tar"

    @property
    def best_path(self) -> Path:
        return self.output_dir / "model_best.pth.tar"

    @property
    def last_path(self) -> Path:
        return self.output_dir / "model_last.pth.tar"

    def epoch_path(self, epoch: int) -> Path:
        return self.output_dir / f"{self.model_name}_epoch{epoch}.pth"

    def _payload(
        self,
        *,
        epoch: int,
        state: TrainingState,
        metrics: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            **{
                name: _state_dict(component)
                for name, component in self.components.items()
            },
            "epoch": int(epoch),
            "best_mAP": float(state.best_mAP),
            "best_epoch": int(state.best_epoch),
            "metrics": dict(metrics or {}),
        }

    def save_epoch(
        self,
        epoch: int,
        state: TrainingState,
        metrics: Mapping[str, Any] | None = None,
    ) -> Path:
        path = self.epoch_path(epoch)
        _atomic_torch_save(
            self._payload(epoch=epoch, state=state, metrics=metrics),
            path,
        )
        _atomic_link_or_copy(path, self.latest_path)
        return path

    def mark_best(self, epoch_path) -> None:
        _atomic_link_or_copy(Path(epoch_path), self.best_path)

    def mark_last(self, epoch_path) -> None:
        _atomic_link_or_copy(Path(epoch_path), self.last_path)

    def resume(
        self,
        path=None,
        *,
        strict_model=True,
    ) -> TrainingState:
        checkpoint_path = Path(path) if path else self.latest_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        missing = [
            name
            for name, component in self.components.items()
            if component is not None and checkpoint.get(name) is None
        ]
        if missing:
            raise ValueError(
                "RESUME_TRAIN requires a complete training checkpoint; "
                f"{checkpoint_path} is missing: {', '.join(missing)}. "
                "Use the model-loading option for weights-only checkpoints."
            )
        for name, component in self.components.items():
            _load_state_dict(
                component,
                checkpoint.get(name),
                strict=strict_model if name in {"model", "center_criterion"} else True,
            )

        return TrainingState(
            epoch=int(checkpoint.get("epoch", 0)),
            best_mAP=float(checkpoint.get("best_mAP", float("-inf"))),
            best_epoch=int(checkpoint.get("best_epoch", 0)),
        )


def period_due(epoch: int, period: int) -> bool:
    return period > 0 and epoch % period == 0


def save_epoch_state(
    *,
    checkpointer,
    state,
    epoch,
    max_epochs,
    checkpoint_period,
    metrics,
    logger: logging.Logger,
):
    """Apply the shared epoch/best/last persistence policy."""

    state.epoch = epoch
    is_best = False
    if metrics is not None:
        is_best = state.update_best(metrics["mAP"], epoch)

    should_save = (
        metrics is not None
        or period_due(epoch, checkpoint_period)
        or epoch == max_epochs
    )
    epoch_path = None
    if should_save:
        epoch_path = checkpointer.save_epoch(epoch, state, metrics)
        logger.info("Saved complete checkpoint: %s", epoch_path)

    if is_best:
        checkpointer.mark_best(epoch_path)
        logger.info(
            "New best target mAP: %.1f%% at epoch %d",
            state.best_mAP * 100,
            state.best_epoch,
        )

    if epoch == max_epochs:
        checkpointer.mark_last(epoch_path)
        logger.info("Saved final checkpoint: %s", checkpointer.last_path)

    return is_best


def save_checkpoint(
    path,
    *,
    model,
    objective,
    optimizer,
    scheduler,
    epoch,
    extra=None,
):
    """Backward-compatible checkpoint writer for the unified Trainer."""

    path = Path(path)
    payload = {
        "model": _state_dict(model),
        "objective": _state_dict(objective),
        "optimizer": _state_dict(optimizer),
        "scheduler": _state_dict(scheduler),
        "epoch": epoch,
        "extra": extra or {},
    }
    _atomic_torch_save(payload, path)
