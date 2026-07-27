from __future__ import print_function, absolute_import

from .build import build_datamanager
from .caption_store import CaptionStore
from .records import ReIDSample, image_only_sample

__all__ = [
    "CaptionStore",
    "Dataset",
    "ImageDataManager",
    "ImageDataset",
    "ReIDSample",
    "VideoDataManager",
    "VideoDataset",
    "build_datamanager",
    "image_only_sample",
    "register_image_dataset",
    "register_video_dataset",
]


def __getattr__(name):
    """Delay legacy dataset imports until an old entry point needs them."""

    if name in {"ImageDataManager", "VideoDataManager"}:
        from . import datamanager

        return getattr(datamanager, name)
    if name in {
        "Dataset",
        "ImageDataset",
        "VideoDataset",
        "register_image_dataset",
        "register_video_dataset",
    }:
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
