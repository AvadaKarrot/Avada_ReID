__all__ = ["make_model", "make_model_debug"]


def __getattr__(name):
    """Load legacy model factories only when they are explicitly requested."""

    if name == "make_model":
        from .make_model import make_model

        return make_model
    if name == "make_model_debug":
        from .make_model_debug import make_model_debug

        return make_model_debug
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
