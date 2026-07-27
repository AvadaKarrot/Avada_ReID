from itertools import chain

import torch


def build_optimizer(cfg, model, objective):
    """Build one optimizer over image-model and trainable objective modules."""

    parameters = [
        parameter
        for parameter in chain(model.parameters(), objective.parameters())
        if parameter.requires_grad
    ]
    name = cfg.SOLVER.OPTIMIZER_NAME.lower()
    kwargs = {
        "lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
    }

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=cfg.SOLVER.MOMENTUM,
            **kwargs,
        )
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")
