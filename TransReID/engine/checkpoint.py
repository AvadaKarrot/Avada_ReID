from pathlib import Path

import torch


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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "objective": objective.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "extra": extra or {},
        },
        path,
    )
