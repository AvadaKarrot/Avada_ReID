"""Compatibility entry point for the unified source-Caption trainer.

The previous script constructed ``make_model_caption.py``, whose target
forward required text. New experiments must use the image-only model plus a
Caption objective, so this filename now delegates to the canonical entry.
"""

import warnings

from tools.train import main


if __name__ == "__main__":
    warnings.warn(
        "train_caption.py is deprecated; use tools/train.py. "
        "The supplied config must use OBJECTIVE.CAPTION, not MODEL.CAPTION.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
