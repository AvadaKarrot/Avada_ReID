__all__ = ["make_model"]


def __getattr__(name):
    """Load legacy model factories only when they are explicitly requested."""

    if name == "make_model":
        from .make_model import make_model

        return make_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
