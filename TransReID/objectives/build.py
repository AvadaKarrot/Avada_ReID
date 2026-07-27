from .reid_objective import ReIDObjective


def _getattr_path(obj, path, default=None):
    current = obj
    for name in path.split("."):
        if not hasattr(current, name):
            return default
        current = getattr(current, name)
    return current


def build_objective(cfg, caption_objective=None) -> ReIDObjective:
    return ReIDObjective(
        triplet_margin=float(_getattr_path(cfg, "SOLVER.MARGIN", 0.3)),
        id_weight=float(_getattr_path(cfg, "MODEL.ID_LOSS_WEIGHT", 1.0)),
        triplet_weight=float(
            _getattr_path(cfg, "MODEL.TRIPLET_LOSS_WEIGHT", 1.0)
        ),
        label_smoothing=float(
            _getattr_path(cfg, "OBJECTIVE.LABEL_SMOOTHING", 0.0)
        ),
        caption_objective=caption_objective,
        caption_weight=float(
            _getattr_path(cfg, "OBJECTIVE.CAPTION.WEIGHT", 0.0)
        ),
    )
