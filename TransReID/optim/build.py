import torch


def build_optimizer(cfg, model, objective):
    """Build one optimizer over image-model and trainable objective modules."""

    parameters = []
    named_parameters = list(model.named_parameters()) + [
        (f"objective.{name}", parameter)
        for name, parameter in objective.named_parameters()
    ]
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        learning_rate = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in name:
            learning_rate *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR and any(
            marker in name for marker in ("classifier", "arcface")
        ):
            learning_rate = cfg.SOLVER.BASE_LR * 2
        parameters.append(
            {
                "params": [parameter],
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
    name = cfg.SOLVER.OPTIMIZER_NAME.lower()

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=cfg.SOLVER.MOMENTUM,
        )
    if name == "adamw":
        return torch.optim.AdamW(parameters)
    if name == "adam":
        return torch.optim.Adam(parameters)
    raise ValueError(f"Unsupported optimizer: {cfg.SOLVER.OPTIMIZER_NAME}")
