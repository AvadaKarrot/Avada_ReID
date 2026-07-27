from typing import Callable, Dict

from torch import nn


BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_backbone(name: str):
    def decorator(factory: Callable[..., nn.Module]):
        if name in BACKBONE_REGISTRY:
            raise KeyError(f"Backbone '{name}' is already registered")
        BACKBONE_REGISTRY[name] = factory
        return factory

    return decorator


def build_backbone(name: str, **kwargs) -> nn.Module:
    try:
        factory = BACKBONE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(BACKBONE_REGISTRY))
        raise KeyError(
            f"Unknown backbone '{name}'. Available backbones: {available}"
        ) from exc
    return factory(**kwargs)
